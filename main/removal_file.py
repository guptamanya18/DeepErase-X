import cv2
import numpy as np
import os
from PIL import Image
from core import process_inpaint


def create_mask(image, final_mask):
    """
    Creates an RGBA mask for inpainting.

    - `image`: Original image (PIL Image).
    - `final_mask`: Binary mask (NumPy array) where 255 = regions to be inpainted.

    Returns:
    - RGBA mask (NumPy array).
    """
    img_array = np.array(image)
    h, w, _ = img_array.shape
    mask = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA mask

    # Detect magenta pixels (255, 0, 255) → Mark as transparent
    magenta_pixels = (img_array[:, :, 0] == 255) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 255)

    # Default: Fully opaque mask
    mask[:, :, :] = [0, 0, 0, 255]

    # Apply transparency where mask is active (255)
    mask[final_mask == 255] = [0, 0, 0, 0]  # Inpaint regions transparent

    # Apply magenta masking
    mask[magenta_pixels] = [0, 0, 0, 0]

    return mask


def apply_inpainting(image, final_mask, output_folder):
    """
    Applies inpainting to an image using the final combined mask.

    - `image`: Original input image (NumPy array).
    - `final_mask`: Final binary mask where 255 = inpainted regions.
    - `output_folder`: Folder to save results.

    Returns:
    - Inpainted image (NumPy array).
    """
    img_input = Image.fromarray(image).convert("RGBA")
    mask = create_mask(img_input, final_mask)  # Generate RGBA mask

    # Save mask for debugging
    mask_output_path = os.path.join(output_folder, "image_mask.png")
    cv2.imwrite(mask_output_path, mask)

    # Process inpainting using `process_inpaint`
    output = process_inpaint(np.array(img_input), mask)

    # Save inpainted output
    inpaint_output_path = os.path.join(output_folder, "image_inpainted.png")
    cv2.imwrite(inpaint_output_path, output)

    return output
