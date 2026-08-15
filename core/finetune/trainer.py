import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from core.finetune.constants import LOG_LEVEL, LOG_NAME
from core.finetune.datasets import WanI2VDatasetWithResize
from core.finetune.datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from core.finetune.schemas import Args, Components, State
from core.finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_memory_statistics,
    get_optimizer,
    string_to_filename,
    unload_model,
    unwrap_model,
)


logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.fsdp import BackwardPrefetch
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.train_resolution[0],
            train_height=self.args.train_resolution[1],
            train_width=self.args.train_resolution[2],
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False) # 원래 True였음
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.accelerator.is_main_process:
            self.args.output_dir = Path(self.args.output_dir)
            self.args.output_dir.mkdir(parents=True, exist_ok=True)

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config
        if self.components.transformer_2 is not None:
            self.state.transformer_2_config = self.components.transformer_2.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing dataset and dataloader")

        if self.args.model_type == "wan-i2v":
            self.dataset = WanI2VDatasetWithResize(
                **(self.args.model_dump()),
                device=self.accelerator.device,
                max_num_frames=self.state.train_frames,
                height=self.state.train_height,
                width=self.state.train_width,
                trainer=self,              
            )
        else:
            raise ValueError(f"Invalid model type: {self.args.model_type}")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(self.accelerator.device, dtype=self.state.weight_dtype)
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        # Precompute latent for video and prompt embedding
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=self.args.pin_memory,
        )
        tmp_data_loader = self.accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tmp_data_loader:
            ...
        self.accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=True,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name in ["transformer", "transformer_2"]:
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            transformer_2_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.components.transformer_2.add_adapter(transformer_2_lora_config)
            # LoRA权重与基座格式对齐
            self.components.transformer = self.components.transformer.to(dtype=torch.bfloat16)
            self.components.transformer_2 = self.components.transformer_2.to(dtype=torch.bfloat16)
            self.__prepare_saving_loading_hooks(transformer_lora_config, transformer_2_lora_config)
            for name, param in self.components.transformer.named_parameters():
                if 'learnable_domain_embeddings' in name:
                    param.requires_grad_(True)
                    logger.info(f"Training transfomer {name} after adding LoRA")
            for name, param in self.components.transformer_2.named_parameters():
                if 'learnable_domain_embeddings' in name:
                    param.requires_grad_(True)
                    logger.info(f"Training transformer_2 {name} after adding LoRA")

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer", "transformer_2"] + self.UNLOAD_LIST
        if self.args.unload_vae_after_precompute:
            ignore_list.append("vae")
            logger.info("Keeping VAE on CPU after latent precompute")
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()
            if self.components.transformer_2 is not None:
                self.components.transformer_2.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        # models_to_cast = [self.components.transformer]
        # if self.components.transformer_2 is not None:
        #    models_to_cast.append(self.components.transformer_2)
        # cast_training_params(models_to_cast, dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(filter(lambda p: p.requires_grad, self.components.transformer.parameters()))
        if self.components.transformer_2 is not None:
            trainable_parameters += list(filter(lambda p: p.requires_grad, self.components.transformer_2.parameters()))
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        if self.components.transformer_2 is not None:
            self.components.transformer, self.components.transformer_2, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.components.transformer_2, self.optimizer, self.data_loader, self.lr_scheduler)
        else:
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler = self.accelerator.prepare(
            self.components.transformer, self.optimizer, self.data_loader, self.lr_scheduler)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name or "finetrainers-experiment"
        # self.accelerator.init_trackers(tracker_name, config=self.args.model_dump())
        config = self.args.model_dump()

        sanitized_config = {
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v
            for k, v in config.items()
        }
        
        self.accelerator.init_trackers(tracker_name, config=sanitized_config)

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        (
            resume_from_checkpoint_path,
            initial_global_step,
            global_step,
            first_epoch,
        ) = get_latest_ckpt_path_to_resume_from(
            resume_from_checkpoint=self.args.resume_from_checkpoint,
            num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
        )
        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            if self.components.transformer_2 is not None:
                self.components.transformer_2.train()
                models_to_accumulate = [self.components.transformer, self.components.transformer_2]
            else:
                models_to_accumulate = [self.components.transformer]
            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norms = [self.components.transformer.get_global_grad_norm()]
                            if self.components.transformer_2 is not None:
                               grad_norms.append(self.components.transformer_2.get_global_grad_norm())
                            grad_norm = max(
                               g.item() if torch.is_tensor(g) else g
                               for g in grad_norms)
                        else:
                            all_transformer_parameters = list(self.components.transformer.parameters())
                            if self.components.transformer_2 is not None:
                                all_transformer_parameters = list(self.components.transformer.parameters()) + list(self.components.transformer_2.parameters())
                            grad_norm = accelerator.clip_grad_norm_(
                                all_transformer_parameters, self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

        accelerator.wait_for_everyone()
        if global_step % self.args.checkpointing_steps == 0:
            logger.info(f"Skipping duplicate final checkpoint save for step {global_step}")
        else:
            self.__maybe_save_checkpoint(global_step, must_save=True)


        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(self.components, name, component.to(self.accelerator.device, dtype=dtype))

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config, transformer_2_lora_config = None):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        """
        register save/load hook。每次 checkpoint 保存时，分别写入 transformer_lora.bin 和
        transformer_2_lora.bin; 恢复时从这两个文件加载。

        Args:
            transformer_lora_config:   transformer 的 LoRA 配置
            transformer_2_lora_config: transformer_2 的 LoRA 配置（可选，默认同 transformer）
        """
        if transformer_2_lora_config is None:
            transformer_2_lora_config = transformer_lora_config

        lora_configs = {
            "transformer": transformer_lora_config,
            "transformer_2": transformer_2_lora_config,
        }
        # save hook
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                # 使用 unwrapped 后的对象身份 (id) 做精确匹配，避免两个同构模型
                # 因 isinstance 无法区分而依赖 models 列表顺序的隐式风险
                transformer_ref = unwrap_model(self.accelerator, self.components.transformer)
                ref_to_key = {id(transformer_ref): "transformer"}

                if self.components.transformer_2 is not None:
                    transformer_2_ref = unwrap_model(self.accelerator, self.components.transformer_2)
                    ref_to_key[id(transformer_2_ref)] = "transformer_2"

                for model in models:
                    unwrapped = unwrap_model(self.accelerator, model)
                    key = ref_to_key.get(id(unwrapped))
                    if key is None:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    lora_state_dict = get_peft_model_state_dict(unwrapped)
                    save_path = os.path.join(output_dir, f"{key}_lora.bin")
                    torch.save(lora_state_dict, save_path)
                    logger.info(f"Saved {key} LoRA weights → {save_path}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()


        def load_model_hook(models, input_dir):
            # 使用 unwrapped 后的对象身份 (id) 做精确匹配，与 save 侧一致，
            # 完全消除对 models 列表顺序的隐式依赖
            transformer_ref = unwrap_model(self.accelerator, self.components.transformer)
            ref_to_key = {id(transformer_ref): "transformer"}

            if self.components.transformer_2 is not None:
                transformer_2_ref = unwrap_model(self.accelerator, self.components.transformer_2)
                ref_to_key[id(transformer_2_ref)] = "transformer_2"

            model_map = {}
            while len(models) > 0:
                model = models.pop()
                unwrapped = unwrap_model(self.accelerator, model)
                key = ref_to_key.get(id(unwrapped))
                if key is None:
                    raise ValueError(f"Unexpected load model: {unwrapped.__class__}")
                if key in model_map:
                    logger.warning(
                        f"Duplicate model identity match for '{key}' — overwriting."
                        " This may indicate the same model was passed twice in models list."
                    )
                model_map[key] = unwrapped

            # 对每个 transformer，从独立 .bin 文件加载 LoRA 权重
            for key in ["transformer", "transformer_2"]:
                if key not in model_map:
                    logger.warning(f"Model {key} not found in checkpoint models list, skipping")
                    continue
                transformer_ = model_map[key]
                save_path = os.path.join(input_dir, f"{key}_lora.bin")
                if not os.path.exists(save_path):
                    logger.warning(f"LoRA weights file not found: {save_path}, skipping {key}")
                    continue
                lora_state_dict = torch.load(save_path, map_location="cpu")
                incompatible_keys = set_peft_model_state_dict(
                    transformer_, lora_state_dict, adapter_name="default"
                )
                if incompatible_keys is not None:
                    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                    if unexpected_keys:
                        logger.warning(
                            f"Loading {key} adapter weights → unexpected keys: {unexpected_keys}"
                        )
                    missing_keys = getattr(incompatible_keys, "missing_keys", None)
                    if missing_keys:
                        logger.warning(
                            f"Loading {key} adapter weights → missing keys: {missing_keys}"
                        )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(self, global_step: int, must_save: bool = False):
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED or self.accelerator.is_main_process:
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.args.output_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)
