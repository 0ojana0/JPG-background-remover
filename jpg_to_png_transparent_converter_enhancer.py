import sys
import subprocess
import importlib.util

def install_and_import(package_name, import_name=None):
    """
    Checks if a Python module is installed; if not, installs it via pip automatically.
    """
    if import_name is None:
        import_name = package_name

    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[*] Package '{package_name}' not found. Installing automatically via pip...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"[+] Successfully installed '{package_name}'.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Failed to install '{package_name}'. Attempting basic installation without flags...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Essential dependencies for image enhancement and AI-assisted background removal
REQUIRED_PACKAGES = [
    ("Pillow", "PIL"),
    ("rembg", "rembg"),
    ("numpy", "numpy"),
    ("opencv-python", "cv2")
]

print("=" * 60)
print("  JPG -> PNG HD CONVERTER (BACKGROUND REMOVAL & 1000x1000 ENHANCEMENT)")
print("=" * 60)
print("[*] Checking system dependencies...")

for pkg, imp in REQUIRED_PACKAGES:
    install_and_import(pkg, imp)

print("[+] All dependencies verified.\n")

import os
import io
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import rembg

def remove_background_rembg(pil_image):
    """
    Removes background using the pre-trained neural network in rembg.
    Handles both light and dark backgrounds, as well as complex silhouettes.
    """
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    input_bytes = img_byte_arr.getvalue()
    
    # Process image with rembg session (Alpha matting enabled for cleaner edges)
    output_bytes = rembg.remove(
        input_bytes,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5
    )
    
    return Image.open(io.BytesIO(output_bytes)).convert("RGBA")

def remove_background_opencv_fallback(pil_image):
    """
    Algorithmic fallback for background removal using OpenCV GrabCut & thresholding
    if neural model is unavailable or encounters memory limits.
    """
    # Convert PIL Image to OpenCV BGR format
    np_img = np.array(pil_image)
    if np_img.ndim == 2:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    elif np_img.shape[2] == 4:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Determine if background is predominantly light or dark using corner samples
    corners = [
        gray[0:max(1, int(h * 0.05)), 0:max(1, int(w * 0.05))],
        gray[0:max(1, int(h * 0.05)), max(0, int(w * 0.95)):w],
        gray[max(0, int(h * 0.95)):h, 0:max(1, int(w * 0.05))],
        gray[max(0, int(h * 0.95)):h, max(0, int(w * 0.95)):w]
    ]
    corner_mean = np.mean([np.mean(c) for c in corners])
    is_light_bg = corner_mean > 127

    # Initialize GrabCut mask
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Create bounding box for GrabCut excluding outer 3% margin
    margin_x = max(1, int(w * 0.03))
    margin_y = max(1, int(h * 0.03))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    try:
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        # Probable and definite foreground are kept
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    except Exception:
        # Fallback to thresholding if GrabCut fails on extreme aspect ratios
        if is_light_bg:
            _, fg_mask = cv2.threshold(gray, int(corner_mean * 0.85), 255, cv2.THRESH_BINARY_INV)
        else:
            _, fg_mask = cv2.threshold(gray, int(corner_mean * 1.15) + 30, 255, cv2.THRESH_BINARY)

    # Smooth the alpha mask edge with Gaussian Blur to avoid harsh jagged cutouts
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)

    # Reassemble as RGBA PIL Image
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack((rgb, fg_mask))
    return Image.fromarray(rgba, mode="RGBA")

def defringe_and_clean_edges(rgba_image):
    """
    Subtly feather and clean semi-transparent halo pixels around edges.
    """
    r, g, b, a = rgba_image.split()
    
    # Erode slight translucent halos
    np_a = np.array(a)
    kernel = np.ones((2, 2), np.uint8)
    eroded_a = cv2.morphologyEx(np_a, cv2.MORPH_CLOSE, kernel)
    cleaned_alpha = Image.fromarray(eroded_a)
    
    return Image.merge("RGBA", (r, g, b, cleaned_alpha))

def enhance_image_clarity(rgba_image):
    """
    Applies multi-stage image clarity improvements:
    - Micro-contrast boost
    - Color saturation adjustment
    - Unsharp Mask for fine textures without introducing alpha artifacts
    """
    r, g, b, a = rgba_image.split()
    rgb_composite = Image.merge("RGB", (r, g, b))

    # 1. Subtle contrast improvement
    contrast_enhancer = ImageEnhance.Contrast(rgb_composite)
    enhanced_rgb = contrast_enhancer.enhance(1.08)

    # 2. Gentle color vibrance improvement
    color_enhancer = ImageEnhance.Color(enhanced_rgb)
    enhanced_rgb = color_enhancer.enhance(1.05)

    # 3. Apply Unsharp Masking for crisp edge details
    sharpened_rgb = enhanced_rgb.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=125, threshold=3)
    )

    r_new, g_new, b_new = sharpened_rgb.split()
    return Image.merge("RGBA", (r_new, g_new, b_new, a))

def resize_and_pad_to_1000(rgba_image, target_size=(1000, 1000), padding_pct=0.06):
    """
    Scales the subject image proportionally to fit within 1000x1000 pixels
    with high-quality Lanczos resampling and centers it onto a transparent canvas.
    """
    target_w, target_h = target_size
    max_w = int(target_w * (1.0 - 2 * padding_pct))
    max_h = int(target_h * (1.0 - 2 * padding_pct))

    # Calculate proportional scale factor
    orig_w, orig_h = rgba_image.size
    ratio = min(max_w / orig_w, max_h / orig_h)
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))

    # High-quality Lanczos downsampling/upsampling
    resized_subject = rgba_image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Create transparent 1000x1000 canvas (RGBA: 0,0,0,0)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))

    # Center the subject onto the canvas
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized_subject, (offset_x, offset_y), mask=resized_subject)

    return canvas

def select_jpg_files():
    """
    Opens an OS-native multi-file picker for selecting JPG / JPEG files.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_types = [
        ("JPEG Images", "*.jpg *.jpeg *.JPG *.JPEG"),
        ("All Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
        ("All Files", "*.*")
    ]

    selected_files = filedialog.askopenfilenames(
        title="Select JPG Images to Convert and Enhance (1000x1000 PNG)",
        filetypes=file_types
    )
    root.destroy()
    return list(selected_files)

def process_single_image(file_path, output_dir):
    """
    Pipeline execution for a single image file.
    """
    filename = os.path.basename(file_path)
    base_name, _ = os.path.splitext(filename)
    output_filename = f"{base_name}_1000x1000_transparent.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n[>] Processing: {filename}")
    print("    - Opening and standardizing color space...")
    with Image.open(file_path) as raw_img:
        # Correct image orientation based on EXIF tag if present
        normalized_img = ImageOps.exif_transpose(raw_img)
        source_img = normalized_img.convert("RGB")

    # Step 1: Remove Light/Dark Background
    print("    - Removing background (light/dark adaptive)...")
    try:
        isolated_img = remove_background_rembg(source_img)
    except Exception as e:
        print(f"    [!] AI background removal fallback triggered: {e}")
        isolated_img = remove_background_opencv_fallback(source_img)

    # Step 2: Edge defringe & cleaning
    print("    - Cleaning edge transparency & defringing...")
    cleaned_img = defringe_and_clean_edges(isolated_img)

    # Step 3: Enhance clarity, contrast, and micro-sharpness
    print("    - Applying clarity, vibrance, and unsharp mask...")
    enhanced_img = enhance_image_clarity(cleaned_img)

    # Step 4: Proportional Lanczos scaling to 1000x1000
    print("    - Resizing and centering on 1000x1000 transparent canvas...")
    final_1000x1000 = resize_and_pad_to_1000(enhanced_img, target_size=(1000, 1000))

    # Step 5: Save as optimized 32-bit PNG
    print(f"    - Saving PNG: {output_filename}")
    final_1000x1000.save(output_path, format="PNG", optimize=True)
    print(f"[✓] Saved successfully -> {output_path}")

    return output_path

def main():
    print("[*] Launching file chooser dialog...")
    files = select_jpg_files()

    if not files:
        print("[!] No files selected. Exiting process.")
        return

    first_dir = os.path.dirname(files[0])
    output_directory = os.path.join(first_dir, "converted_png_1000x1000")
    os.makedirs(output_directory, exist_ok=True)

    print(f"\n[+] Selected {len(files)} file(s).")
    print(f"[+] Output folder: {output_directory}")

    successful = 0
    failed = 0

    for idx, file_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] ----------------------------------------")
        try:
            process_single_image(file_path, output_directory)
            successful += 1
        except Exception as err:
            print(f"[!] Error processing {os.path.basename(file_path)}: {err}")
            failed += 1

    # Final summary display
    print("\n" + "=" * 60)
    print(f"PROCESSING COMPLETED")
    print(f"  Total Processed: {len(files)}")
    print(f"  Successful:      {successful}")
    print(f"  Failed:          {failed}")
    print(f"  Output Directory: {output_directory}")
    print("=" * 60)

    # Graphical notification
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Conversion Completed",
        f"Processed {successful} image(s) to 1000x1000 transparent PNGs.\n\nSaved in:\n{output_directory}"
    )
    root.destroy()

if __name__ == "__main__":
    main()