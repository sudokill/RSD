import os
import yaml
import shutil
import subprocess
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import sys
from tqdm import tqdm

# This is the same complete config structure from video_fixer.py
BASE_CONFIG_STRUCTURE = {
    'model': {
        'dim': 64, 
        'dim_mults': [1, 2, 4, 8], 
        'num_classes': 1, 
        'cond_drop_prob': 0
    },
    'diffusion': {
        'image_size': 128,
        'timesteps': 1000, 
        'sampling_timesteps': 8, 
        'beta_schedule': 'linear', 
        'objective': 'pred_x0'
    },
    'dataset': {
        'train_folder': 'RS_Real/train',
        'test_folder': '', 
        'image_size': 128,
        'augment_horizontal_flip': False
    },
    'trainer': {
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
        'data_root': '', 
        'save_path': '', 
        'checkpoint': "./checkpoint/RS_Real.pt"
    }
}


def prepare_image_for_model(image_path, output_dir, model_size):
    """Pre-processes a single image to be model-ready."""
    try:
        # Pre-process the image (resize with padding)
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((model_size, model_size))
        
        padded_img = Image.new("RGB", (model_size, model_size), (0, 0, 0))
        paste_pos = ((model_size - img.width) // 2, (model_size - img.height) // 2)
        padded_img.paste(img, paste_pos)

        base_name = Path(image_path).stem
        processed_img_path = output_dir / f"{base_name}.png"
        npz_path = output_dir / f"{base_name}.npz"
        
        padded_img.save(processed_img_path)

        condition_array = np.array(padded_img, dtype=np.uint8)
        gs_array = np.zeros_like(condition_array)
        out_flow_array = np.zeros((2, model_size, model_size), dtype=np.float32)

        np.savez(npz_path, condition=condition_array, gs=gs_array, out_flow=out_flow_array)
        return True
    except Exception as e:
        print(f"Could not prepare image {image_path}: {e}")
        return False

def postprocess_image(model_output_path, original_dims, final_image_path):
    """Takes the model's square output and resizes it back to original dimensions."""
    try:
        model_img = Image.open(model_output_path).convert("RGB")
        
        original_width, original_height = original_dims
        ratio = min(model_img.width / original_width, model_img.height / original_height)
        thumb_width, thumb_height = int(original_width * ratio), int(original_height * ratio)

        crop_pos_x = (model_img.width - thumb_width) // 2
        crop_pos_y = (model_img.height - thumb_height) // 2
        crop_box = (crop_pos_x, crop_pos_y, crop_pos_x + thumb_width, crop_pos_y + thumb_height)
        cropped_img = model_img.crop(crop_box)
        
        final_img = cropped_img.resize(original_dims, Image.LANCZOS)
        final_img.save(final_image_path)
        return True
    except Exception as e:
        print(f"Could not post-process image {model_output_path}: {e}")
        return False

def main(args):
    """Main pipeline for a single image."""
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    workspace = Path(f"./temp_workspace_{input_path.stem}")
    if workspace.exists(): shutil.rmtree(workspace)
    workspace.mkdir()

    try:
        preprocessed_dir = workspace / "preprocessed"
        model_output_dir = workspace / "model_output"
        preprocessed_dir.mkdir()
        model_output_dir.mkdir()

        print(f"--- Step 1: Pre-processing image for model size {args.model_size}x{args.model_size} ---")
        original_image = Image.open(input_path)
        original_dims = original_image.size
        prepare_image_for_model(input_path, preprocessed_dir, args.model_size)

        print("--- Step 2: Running RS-Diffusion model ---")
        temp_config_path = workspace / "temp_config.yaml"
        
        config = BASE_CONFIG_STRUCTURE
        config['diffusion']['image_size'] = args.model_size
        config['dataset']['image_size'] = args.model_size
        config['dataset']['test_folder'] = str(preprocessed_dir.resolve())
        config['test']['save_path'] = str(model_output_dir.resolve())
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        subprocess.run([
            sys.executable, "./sample_RS_real.py", 
            "--config", str(temp_config_path.resolve())
        ], check=True)
        print("Model processing complete.")

        print("--- Step 3: Post-processing final image ---")
        # The model saves corrected images in an 'image' subdirectory
        image_subdir = model_output_dir / 'image'
        # The output file will have the same stem as the preprocessed file
        output_file_stem = Path(sorted(preprocessed_dir.glob("*.png"))[0]).stem
        model_output_file = image_subdir / f"{output_file_stem}.jpg"
        
        postprocess_image(model_output_file, original_dims, Path(args.output_path))
        
        print(f"\nSUCCESS! Corrected image saved to {args.output_path}")

    finally:
        if not args.keep_temp and workspace.exists():
            print("--- Cleaning up temporary files ---")
            shutil.rmtree(workspace)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end rolling shutter correction for a single image.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input image file (e.g., .jpg, .png).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the corrected output image.")
    parser.add_argument("--model_size", type=int, default=128, help="Processing size for the model. Larger is higher quality but slower.")
    parser.add_argument("--keep_temp", action='store_true', help="If set, keeps the temporary workspace folder for debugging.")
    
    args = parser.parse_args()
    main(args)