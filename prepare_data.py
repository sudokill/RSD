import numpy as np
from PIL import Image, ImageOps
import os
import argparse

def prepare_image_for_model(original_path, output_dir, model_size=128):
    """
    Takes an original image, preprocesses it to a padded square of model_size,
    and saves both the processed PNG and the corresponding .npz file.
    """
    if not os.path.exists(original_path):
        print(f"Error: Original image not found at {original_path}")
        return

    print(f"--- Preparing {original_path} for model size {model_size}x{model_size} ---")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Pre-process the image (resize with padding)
    img = Image.open(original_path).convert("RGB")
    img.thumbnail((model_size, model_size)) # Resize while maintaining aspect ratio
    
    padded_img = Image.new("RGB", (model_size, model_size), (0, 0, 0))
    paste_x = (model_size - img.width) // 2
    paste_y = (model_size - img.height) // 2
    padded_img.paste(img, (paste_x, paste_y))

    # Define paths for the new processed image and the npz file
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    processed_img_name = f"{base_name}_{model_size}px.png"
    processed_img_path = os.path.join(output_dir, processed_img_name)
    npz_path = os.path.join(output_dir, f"{base_name}_{model_size}px.npz")

    # Save the processed image
    padded_img.save(processed_img_path)
    print(f"Saved processed image to: {processed_img_path}")
    
    # Create correctly-sized arrays for the .npz file
    condition_array = np.array(padded_img, dtype=np.uint8)
    gs_array = np.zeros_like(condition_array)
    out_flow_array = np.zeros((2, model_size, model_size), dtype=np.float32)

    # Save the .npz file
    np.savez(npz_path, condition=condition_array, gs=gs_array, out_flow=out_flow_array)
    print(f"Successfully created data file at: {npz_path}")
    print("--- Preparation complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for RS-Diffusion.")
    parser.add_argument("input_file", type=str, help="Path to the original image.")
    parser.add_argument("--output_dir", type=str, default="input", help="Directory to save the processed files.")
    parser.add_argument("--size", type=int, default=128, help="The target image_size for the model.")
    args = parser.parse_args()
    
    prepare_image_for_model(args.input_file, args.output_dir, args.size)