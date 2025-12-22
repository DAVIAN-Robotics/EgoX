#!/usr/bin/env python3
"""
Extract captions from exo_GT and ego_GT images using GPT-4o.
Reads paths from top_ego_prior_datasets.json and processes 4 evenly distributed images per task.
"""

import os
import base64
import json
import cv2
import argparse
import copy
from pathlib import Path
from openai import OpenAI
import numpy as np

# Initialize OpenAI client with Letsur API
client = OpenAI(
    base_url="YOUR_BASE_URL",
    api_key="YOUR_API_KEY"
)

# System prompt adapted from captioning.py
system_prompt = (
    "You are a hyper-realistic scene reconstruction AI. Your task is to analyze a sequence of video frames "
    "provided in chronological order and produce a comprehensive, two-part analysis: a static scene overview "
    "followed by a dynamic, frame-by-frame action breakdown. Your guiding principle is **strict objectivity**."
    "\n\n"
    "--- MISSION PROTOCOL ---"
    "\n\n"
    "**Phase 1: Scene Establishment**\n"
    "First, analyze all provided frames to establish a detailed, static description of the physical environment. "
    "Detail the surfaces (walls, floors), furniture, and all unmoving background items. This is your 'establishing shot'."
    "\n\n"
    "**Phase 2: Action Transition Analysis**\n"
    "After establishing the scene, provide a detailed description of the action progression and transitions observed across the sequence. "
    "Focus on how actions evolve, change, and flow from one moment to the next, maintaining awareness of the overall context established in Phase 1."
    "\n\n"
    "--- CRITICAL DIRECTIVES ---"
    "\n\n"
    "**1. Exhaustive Object Inventory: THIS IS YOUR MOST IMPORTANT TASK.**\n"
    "You must meticulously identify and catalog EVERY visible item.\n"
    "- **NO GENERIC TERMS**: Do not use vague words like 'tool', 'box', 'utensil', or 'device'.\n"
    "- **BE SPECIFIC**: Use precise names (e.g., 'smartphone', 'coffee mug', 'wooden spoon', 'cutting board', 'refrigerator', "
    "'laptop computer', 'ceramic bowl', 'stainless steel knife').\n"
    "- **DESCRIBE PROPERTIES**: Include colors, materials, textures, and positions (e.g., 'a blue ceramic mug on a granite countertop')."
    "\n\n"
    "**2. Focus on Hand-Object Interaction: THE ACTION'S CORE.**\n"
    "**- For the `[Exo view]`, your primary narrative focus MUST be the person's hands.** Describe their precise posture, movement, and interaction with objects (e.g., 'the person's right hand grasps the knife handle,' 'the left hand's fingertips stabilize the tomato').\n"
    "- Every action description should revolve around what the hands are doing."
    "\n\n"
    "**3. Strict Objectivity: DESCRIBE, DO NOT INTERPRET.**\n"
    "- **AVOID JUDGMENT**: Do not use subjective or abstract adjectives (e.g., AVOID 'modern', 'beautiful', 'cluttered', 'well-lit'). "
    "Describe only physical, measurable attributes."
    "\n\n"
    "**3. Transition-Focused Analysis**\n"
    "- Analyze the sequence as a continuous flow of actions\n"
    "- Describe how movements and interactions transition and evolve\n"
    "- Focus on the progression and changes rather than individual frame descriptions\n"
    "- Maintain narrative continuity throughout the sequence"
    "\n\n"
    "--- OUTPUT STRUCTURE ---"
    "\n\n"
    "You MUST follow this exact two-block format:"
    "\n\n"
    "[Exo view]\n"
    "**Scene Overview:**\n"
    "<Detailed description of the static background environment from the third-person perspective. List all background objects.>\n"
    "\n"
    "**Action Analysis:**\n"
    "<Describe the progression of actions and transitions observed throughout the sequence. Focus on how movements evolve, "
    "interactions change, and the flow of activities from beginning to end. Describe the continuous narrative of what is happening.>\n"
    "\n\n"
    "[Ego view]\n"
    "**Scene Overview:**\n"
    "<<Detailed description of the environment as it would appear from the subject’s first-person perspective. Anchor every observation in the exocentric evidence.>\n"
    "\n"
    "**Action Analysis:**\n"
    "<Describe the progression of actions and transitions observed throughout the sequence (or the current action for a single image) from the inferred subject’s viewpoint. Focus on how the subject's hands, any held objects, and the surrounding scene would appear, and how the scene evolves from moment to moment.>"
)


def encode_image(image_path):
    """지정된 경로의 이미지를 base64로 인코딩합니다."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def encode_frame_to_base64(frame):
    """Convert OpenCV frame to base64 encoded JPEG with error handling"""
    try:
        if frame is None or frame.size == 0:
            print("Warning: Empty or None frame")
            return None
            
        # Encode with higher quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            print("Warning: Failed to encode frame to JPEG")
            return None
            
        if len(buffer) == 0:
            print("Warning: Empty buffer after encoding")
            return None
            
        return base64.b64encode(buffer).decode()
        
    except Exception as e:
        print(f"Error in encode_frame_to_base64: {e}")
        return None


def get_frame_files(folder_path, num_frames=4):
    """
    Get evenly distributed frame files from a folder.
    First checks if the folder has exactly 49 images.
    
    Args:
        folder_path: Path to the folder containing frame images
        num_frames: Number of frames to extract (default: 4)
        
    Returns:
        List of frame file paths, or empty list if not exactly 49 images
    """
    if not os.path.exists(folder_path):
        print(f"Warning: Folder not found: {folder_path}")
        return []
    
    # Get all frame files
    frame_files = []
    for file in os.listdir(folder_path):
        # if file.startswith("frame_") and file.endswith(".png"):
        if file.endswith(".png"):
            frame_files.append(os.path.join(folder_path, file))
    
    if not frame_files:
        print(f"Warning: No frame files found in {folder_path}")
        return []
    
    # Check if we have exactly 49 images (frame_000000.png to frame_000048.png)
    if len(frame_files) != 49:
        print(f"Warning: Expected 49 images, found {len(frame_files)} in {folder_path}")
        return []
    
    # Sort frame files by frame number
    frame_files.sort()
    
    # Verify we have the correct frame range (000000 to 000048)
    # expected_frames = [f"frame_{i:06d}.png" for i in range(49)]
    expected_frames = [f"{i+1:06d}.png" for i in range(49)]

    actual_frames = [os.path.basename(f) for f in frame_files]
    
    if actual_frames != expected_frames:
        print(f"Warning: Frame sequence incomplete in {folder_path}")
        print(f"Expected: {expected_frames[:5]}...{expected_frames[-5:]}")
        print(f"Actual: {actual_frames[:5]}...{actual_frames[-5:]}")
        return []
    
    # If we have fewer frames than requested, return all
    if len(frame_files) <= num_frames:
        return frame_files
    
    # Select evenly distributed frames
    indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
    selected_frames = [frame_files[i] for i in indices]
    
    return selected_frames


def extract_frames_from_video(video_path, num_frames=4):
    """
    Extract evenly distributed frames from a video file.
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        
    Returns:
        List of cv2 images (numpy arrays)
    """
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return []
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has 0 frames")
            cap.release()
            return []
            
        # Calculate indices for evenly distributed frames
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {idx} from {video_path}")
                
        cap.release()
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []


def load_image(image_path):
    """Load image using OpenCV"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def extract_captions_for_task(task_path, num_frames=4):
    """
    Extract captions for a single task using exo_GT and ego_GT images.
    
    Args:
        task_path: Path to the task folder
        num_frames: Number of frames to process per view (default: 4)
        
    Returns:
        Dictionary with caption results or None if failed
    """
    print(f"Processing task: {task_path}")
    
    # exo_gt_path = os.path.join(task_path, "exo_GT")
    exo_gt_path = task_path
    ego_gt_path = os.path.join(task_path, "ego_GT")
    
    # Check if exo folder exists
    if not os.path.exists(exo_gt_path):
        print(f"Warning: exo_GT folder not found: {exo_gt_path}")
        return None
    
    # Get frame files (must have exactly 49 images each)
    exo_frames = get_frame_files(exo_gt_path, num_frames)
    if not exo_frames:
        print(f"Warning: Skipping {task_path} - missing or incomplete exo sequence")
        print(f"  exo_GT: {len(exo_frames)} frames found")
        return None
    
    ego_frames = []
    if os.path.exists(ego_gt_path):
        ego_frames = get_frame_files(ego_gt_path, num_frames)
        if not ego_frames:
            print(f"Warning: Incomplete or missing ego sequence in {ego_gt_path}. Proceeding with inferred perspective.")
        else:
            print(f"✓ Valid sequence found: {len(exo_frames)} exo frames and {len(ego_frames)} ego frames")
    else:
        print(f"✓ Valid exo sequence found: {len(exo_frames)} frames. No ego frames supplied; inferring perspective.")
    
    # Prepare content for GPT-4o
    content = [
        {
            "type": "text",
            "text": (
                f"Analyze this sequence of {len(exo_frames)} video frames in chronological order. "
                + (
                    "Each frame shows both an exocentric (third-person) and egocentric (first-person) view side by side. "
                    "Provide the scene overview and frame-by-frame action analysis as specified in your instructions."
                    if ego_frames
                    else "Only the exocentric (third-person) view is provided. "
                    "Infer the first-person experience and deliver the scene overview and action analysis exactly as instructed."
                )
            )
        }
    ]
    
    # Process frames in pairs (exo + ego)
    for i in range(len(exo_frames)):
        # Load exo frame
        exo_image = load_image(exo_frames[i])
        if exo_image is None:
            continue
            
        if ego_frames:
            # Load ego frame
            ego_image = load_image(ego_frames[i])
            if ego_image is None:
                continue
            
            # Create side-by-side image and encode in-memory to avoid temp file collisions
            # Resize images to have the same height before concatenating
            exo_height, exo_width = exo_image.shape[:2]
            ego_height, ego_width = ego_image.shape[:2]
            
            if exo_height != ego_height:
                # Resize ego image to match exo image height
                ego_image_resized = cv2.resize(ego_image, (int(ego_width * exo_height / ego_height), exo_height))
                combined_image = np.hstack([exo_image, ego_image_resized])
            else:
                combined_image = np.hstack([exo_image, ego_image])
            
            frame_b64 = encode_frame_to_base64(combined_image)
        else:
            frame_b64 = encode_frame_to_base64(exo_image)
        
        if not frame_b64:
            print(f"Warning: Failed to encode frame {i}")
            continue
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })
        
    
    if len(content) == 1:  # Only text, no images
        print("Error: No frames were successfully processed")
        return None
    
    try:
        print(f"Sending {len(content)-1} images to GPT-4o...")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=4096,
            temperature=0.1
        )
        
        caption = response.choices[0].message.content.strip()
        
        # Validate response
        if not caption or len(caption) < 50:
            print(f"Warning: Received unusually short response: {caption[:100]}...")
            return None
        
        if "I'm unable to" in caption or "I cannot" in caption:
            print(f"Warning: GPT-4o declined to process: {caption[:100]}...")
            return None
        
        # Save caption to task folder
        caption_file = os.path.join(task_path, "caption.json")
        caption_data = {
            "task_path": task_path,
            "exo_frames": exo_frames,
            "ego_frames": ego_frames,
            "num_frames_processed": len(content) - 1,
            "caption": caption
        }
        
        try:
            with open(caption_file, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2, ensure_ascii=False)
            print(f"Caption saved to: {caption_file}")
        except Exception as e:
            print(f"Error saving caption to {caption_file}: {e}")
        
        return caption_data
        
    except Exception as e:
        print(f"Error processing task {task_path}: {e}")
        return None


def extract_caption_from_video(exo_path, ego_path=None, num_frames=4):
    """
    Generate caption from video files directly.
    
    Args:
        exo_path: Path to the exocentric video file
        ego_path: Path to the egocentric video file (optional)
        num_frames: Number of frames to process
        
    Returns:
        Generated caption string or None
    """
    print(f"Processing video: {exo_path}")
    
    # Extract frames
    exo_images = extract_frames_from_video(exo_path, num_frames)
    if not exo_images:
        print(f"Warning: No frames extracted from {exo_path}")
        return None
        
    ego_images = []
    if ego_path and os.path.exists(ego_path):
        ego_images = extract_frames_from_video(ego_path, num_frames)
    
    # Prepare content for GPT-4o
    content = [
        {
            "type": "text",
            "text": (
                f"Analyze this sequence of {len(exo_images)} video frames in chronological order. "
                + (
                    "Each frame shows both an exocentric (third-person) and egocentric (first-person) view side by side. "
                    "Provide the scene overview and frame-by-frame action breakdown. "
                    "Only the prompt itself is needed, without any additional text." # Modified prompt instruction slightly? Or keep standard?
                    # User requested: "extract prompt and change". 
                    # Existing system prompt is quite specific about output format.
                    # I will stick to the system prompt guidance but ensure we pass the right context.
                    "Provide the scene overview and frame-by-frame action analysis as specified in your instructions."
                    if ego_images
                    else "Only the exocentric (third-person) view is provided. "
                    "Infer the first-person experience and deliver the scene overview and action analysis exactly as instructed."
                )
            )
        }
    ]
    
    # Process frames
    for i in range(len(exo_images)):
        exo_image = exo_images[i]
        
        if ego_images and i < len(ego_images):
            ego_image = ego_images[i]
            
            # Resize and combine
            exo_height, exo_width = exo_image.shape[:2]
            ego_height, ego_width = ego_image.shape[:2]
            
            if exo_height != ego_height:
                ego_image_resized = cv2.resize(ego_image, (int(ego_width * exo_height / ego_height), exo_height))
                combined_image = np.hstack([exo_image, ego_image_resized])
            else:
                combined_image = np.hstack([exo_image, ego_image])
            
            frame_b64 = encode_frame_to_base64(combined_image)
        else:
            frame_b64 = encode_frame_to_base64(exo_image)
            
        if not frame_b64:
            continue
            
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}"
            }
        })
        
    if len(content) == 1:
        return None
        
    try:
        print(f"Sending {len(content)-1} images to GPT-4o...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=4096,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def filter_tasks_without_captions(datasets, overwrite=False):
    """
    Filter datasets to only include tasks that don't have caption.json files.
    
    Args:
        datasets: List of dataset dictionaries with "path" key
        overwrite: If True, include all tasks (don't filter)
        
    Returns:
        Filtered list of datasets without captions
    """
    if overwrite:
        print("Overwrite mode: Will process all tasks (including those with existing captions)")
        return datasets
    
    tasks_without_captions = []
    tasks_with_captions = 0
    
    print("Scanning tasks for missing captions...")
    for i, dataset in enumerate(datasets):
        task_path = dataset["path"]
        caption_file = os.path.join(task_path, "caption.json")
        
        if not os.path.exists(caption_file):
            tasks_without_captions.append(dataset)
        else:
            tasks_with_captions += 1
        
        # Progress update every 500 tasks
        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{len(datasets)} tasks... ({len(tasks_without_captions)} without captions, {tasks_with_captions} with captions)")
    
    print(f"\nCaption status:")
    print(f"  Tasks with captions: {tasks_with_captions}")
    print(f"  Tasks without captions: {len(tasks_without_captions)}")
    print(f"  Total tasks: {len(datasets)}")
    
    return tasks_without_captions


def process_all_tasks(json_file_path, output_json_path, max_tasks=None, start_from=None, stop_count=-1, overwrite=False):
    """
    Process tasks from JSON and update the JSON with new prompts.
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file_path}: {e}")
        return

    # Check for different dataset keys
    dataset_keys = ["datasets", "train_datasets", "val_datasets", "test_datasets"]
    all_datasets = []
    
    # We need to track where each dataset came from to update the original structure
    # But for simplicity, we can just iterate over the keys present in data
    
    total_processed = 0
    total_successful = 0
    
    for key in dataset_keys:
        if key not in data:
            continue
            
        datasets = data[key]
        print(f"Processing {len(datasets)} tasks in section '{key}'")
        
        for i, dataset in enumerate(datasets):
            if start_from and total_processed < start_from:
                total_processed += 1
                continue
                
            if max_tasks and total_successful >= max_tasks:
                break
            
            # Identify video paths
            # The meta.json uses 'exo_path' and 'ego_prior_path'
            exo_path = dataset.get("exo_path")
            ego_path = dataset.get("ego_prior_path")
            current_prompt = dataset.get("prompt", "")
            
            if not exo_path:
                print(f"Skipping task {i}: No exo_path found")
                continue
                
            # If prompt exists and not overwrite, skip
            if current_prompt and not overwrite and len(current_prompt) > 100:
                print(f"Skipping task {i}: Prompt already exists")
                continue
                
            # Resolve relative paths
            # Assuming paths in json are relative to the json file location or CWD?
            # User said "example/in_the_wild/videos/..." in the request, and json has "./example/..."
            # Adjust path if needed.
            if exo_path.startswith("./"):
                # exo_path = exo_path[2:] # Remove ./
                pass
            
            # Processing
            caption = extract_caption_from_video(exo_path, ego_path=ego_path)
            
            if caption:
                dataset["prompt"] = caption
                total_successful += 1
                print(f"✓ Updated prompt for {os.path.basename(exo_path)}")
                
                # Intermediate save (optional, but good for long running processes)
                if total_successful % 5 == 0:
                     with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"Progress saved to {output_json_path}")
            else:
                 print(f"✗ Failed to generate caption for {os.path.basename(exo_path)}")
            
            total_processed += 1
            
            if stop_count > 0 and total_successful % stop_count == 0:
                print(f"Paused after {stop_count} successful tasks. Press Enter to continue...")
                input()
                
        if max_tasks and total_successful >= max_tasks:
            break
            
    # Final save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nProcessing complete. Updated {total_successful} tasks.")
    print(f"Saved result to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract captions from exo_GT and ego_GT images")
    parser.add_argument("--json_file", "-j", 
                       default="./example/in_the_wild/meta.json",
                       help="Path to JSON file containing task paths")
    parser.add_argument("--output_json", "-o",
                       default="./example/in_the_wild/meta_updated.json",
                       help="Path to output updated JSON file")
    parser.add_argument("--max_tasks", "-m", type=int,
                       help="Maximum number of tasks to process")
    parser.add_argument("--start_from", "-s", type=int,
                       help="Task index to start from (for resuming)")
    parser.add_argument("--num_frames", "-n", type=int, default=4,
                       help="Number of frames to process per view (default: 4)")
    parser.add_argument("--stop_count", "-sc", type=int, default=-1,
                       help="Number of successful tasks after which to pause (-1 for no pause, default: -1)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Reprocess tasks that already have captions (overwrite existing caption.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    process_all_tasks(
        json_file_path=args.json_file,
        output_json_path=args.output_json,
        max_tasks=args.max_tasks,
        start_from=args.start_from,
        stop_count=args.stop_count,
        overwrite=args.overwrite
    )


if __name__ == "__main__":
    main()
