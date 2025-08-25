import sys
import cv2  # OpenCV for video processing
import yaml
import shutil
import subprocess
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# This dictionary defines the base structure of the config file we will generate.
# It sets the model size and paths dynamically.
BASE_CONFIG_STRUCTURE = {
    'model': {
        'dim': 64, 
        'dim_mults': [1, 2, 4, 8], 
        'num_classes': 1, 
        'cond_drop_prob': 0
    },
    'diffusion': {
        'image_size': 128,  # This will be updated dynamically
        'timesteps': 1000, 
        'sampling_timesteps': 8, 
        'beta_schedule': 'linear', 
        'objective': 'pred_x0'
    },
    'dataset': {
        'train_folder': 'RS_Real/train', # Placeholder, not used by sample.py
        'test_folder': '', # This will be updated dynamically
        'image_size': 128, # This will be updated dynamically
        'augment_horizontal_flip': False
    },
    'trainer': {
        # This whole section is mostly for training, but sample.py might read it.
        # It's safer to include it with placeholder values.
        'train_batch_size': 16,
        'test_batch_size': 1,
        'train_lr': 0.0001,
        'train_num_steps': 150000,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'amp': False,
        'save_and_sample_every': 5000,
        'results_folder': 'result_RS_Real',
        'log_path': 'RS_Real'
    },
    'test': {
        'dataset_name': "RS_Real",
        'data_root': '', # This will be updated dynamically
        'save_path': '', # This will be updated dynamically
        'checkpoint': "./checkpoint/RS_Real.pt"
    }
}

# --- Helper Functions ---

def prepare_frame_for_model(frame_path, output_dir, model_size):
    """
    Pre-processes a single frame to be model-ready.
    1. Resizes with padding to a square.
    2. Creates the corresponding .npz file.
    """
    try:
        # Pre-process the image (resize with padding)
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((model_size, model_size))
        
        padded_img = Image.new("RGB", (model_size, model_size), (0, 0, 0))
        paste_pos = ((model_size - img.width) // 2, (model_size - img.height) // 2)
        padded_img.paste(img, paste_pos)

        base_name = Path(frame_path).stem
        processed_img_path = output_dir / f"{base_name}.png"
        npz_path = output_dir / f"{base_name}.npz"
        
        padded_img.save(processed_img_path)

        # Create correctly-sized arrays for the .npz file
        condition_array = np.array(padded_img, dtype=np.uint8)
        gs_array = np.zeros_like(condition_array)
        out_flow_array = np.zeros((2, model_size, model_size), dtype=np.float32)

        np.savez(npz_path, condition=condition_array, gs=gs_array, out_flow=out_flow_array)
        return True
    except Exception as e:
        print(f"Could not prepare frame {frame_path}: {e}")
        return False

def postprocess_frame(model_output_path, original_dims, final_frame_path):
    """
    Takes the model's square output and resizes it back to original video dimensions.
    """
    try:
        model_img = Image.open(model_output_path).convert("RGB")
        
        # Calculate the aspect ratio of the original to find the un-padded size
        original_width, original_height = original_dims
        ratio = min(model_img.width / original_width, model_img.height / original_height)
        thumb_width, thumb_height = int(original_width * ratio), int(original_height * ratio)

        # Crop out the black bars
        crop_pos_x = (model_img.width - thumb_width) // 2
        crop_pos_y = (model_img.height - thumb_height) // 2
        crop_box = (crop_pos_x, crop_pos_y, crop_pos_x + thumb_width, crop_pos_y + thumb_height)
        cropped_img = model_img.crop(crop_box)
        
        # Resize back to the original frame dimensions
        final_img = cropped_img.resize(original_dims, Image.LANCZOS)
        final_img.save(final_frame_path)
        return True
    except Exception as e:
        print(f"Could not post-process frame {model_output_path}: {e}")
        return False

# --- Main Pipeline Functions ---

def extract_frames(video_path, output_folder):
    print(f"--- Step 1: Extracting frames from {video_path} ---")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    original_dims = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(frame_count), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_folder / f"frame_{i:05d}.png"), frame)
    
    cap.release()
    print("Frame extraction complete.")
    return frame_rate, original_dims

def assemble_video(frames_folder, output_video_path, frame_rate, frame_size):
    print(f"--- Step 5: Assembling video to {output_video_path} ---")
    frame_files = sorted(frames_folder.glob("*.png"))
    
    # Use 'mp4v' codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, frame_rate, frame_size)

    for frame_file in tqdm(frame_files, desc="Assembling video"):
        frame = cv2.imread(str(frame_file))
        writer.write(frame)
        
    writer.release()
    print("Video assembly complete.")

def main(args):
    """Main pipeline orchestrator."""
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return

    # Create a temporary workspace for all intermediate files
    workspace = Path(f"./temp_workspace_{video_path.stem}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        # Define paths for all our stages
        raw_frames_dir = workspace / "1_raw_frames"
        preprocessed_dir = workspace / "2_preprocessed_frames"
        model_output_dir = workspace / "3_model_output"
        final_frames_dir = workspace / "4_final_frames"
        preprocessed_dir.mkdir()
        
        # Make sure the model output and final frames directories exist
        model_output_dir.mkdir()
        final_frames_dir.mkdir()

        # Step 1: Extract frames from video
        frame_rate, original_dims = extract_frames(video_path, raw_frames_dir)

        # Step 2: Pre-process each frame for the model
        print(f"--- Step 2: Pre-processing frames for model size {args.model_size}x{args.model_size} ---")
        raw_frame_files = sorted(raw_frames_dir.glob("*.png"))
        for frame_file in tqdm(raw_frame_files, desc="Preprocessing frames"):
            prepare_frame_for_model(frame_file, preprocessed_dir, args.model_size)

        # Step 3: Run the RS-Diffusion model
        print("--- Step 3: Running RS-Diffusion model ---")
        temp_config_path = workspace / "temp_config.yaml"
        
        # Create the dynamic config file
        config = BASE_CONFIG_STRUCTURE
        config['diffusion']['image_size'] = args.model_size
        config['dataset']['image_size'] = args.model_size
        config['dataset']['test_folder'] = str(preprocessed_dir.resolve())
        config['test']['save_path'] = str(model_output_dir.resolve())
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        # Call the model script as a subprocess
        subprocess.run([
            sys.executable, "./sample_RS_real.py", 
            "--config", str(temp_config_path.resolve())
        ], check=True)

        # Step 4: Post-process frames back to original size
        print("--- Step 4: Post-processing frames ---")
        image_subdir = model_output_dir / 'image'
        model_output_files = sorted(image_subdir.glob("*.jpg")) 
        for model_file in tqdm(model_output_files, desc="Postprocessing frames"):
            final_frame_path = final_frames_dir / f"{model_file.stem}.png"
            postprocess_frame(model_file, original_dims, final_frame_path)
        
        # Step 5: Assemble frames into final video
        assemble_video(final_frames_dir, Path(args.output_path), frame_rate, original_dims)

        print(f"\nSUCCESS! Corrected video saved to {args.output_path}")

    finally:
        # Step 6: Clean up temporary files
        if not args.keep_frames and workspace.exists():
            print("--- Cleaning up temporary files ---")
            shutil.rmtree(workspace)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end rolling shutter correction for videos.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the corrected output video.")
    parser.add_argument("--model_size", type=int, default=128, help="Processing size for the model (e.g., 64, 128). Larger is higher quality but much slower on CPU.")
    parser.add_argument("--keep_frames", action='store_true', help="If set, keeps the temporary workspace folder for debugging.")
    
    args = parser.parse_args()
    main(args)