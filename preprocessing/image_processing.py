from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def resize_images(
    input_dir: str = "data/images",
    size: int | tuple[int, int] = (224, 224),
    output_dir: str = "data/images",
) -> str:
    """
    Resize all images in input_dir and save them to output_dir.

    Args:
        input_dir: Directory containing source images.
        size: Target size as an int (square) or (height, width) tuple.
        output_dir: Directory to save resized images. Defaults to
                    '<input_dir>_resized'.

    Returns:
        Path to the output directory as a string.
    """
    if isinstance(size, int):
        h, w = size, size
    else:
        h, w = int(size[0]), int(size[1])

    input_path = Path(input_dir)
    out_path = Path(output_dir) if output_dir else input_path.parent / (input_path.name + "_resized")
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print(f"No images found in: {input_dir}")
        return str(out_path)

    print(f"Resizing {len(image_files)} images to ({h}, {w}) -> {out_path}")
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path.name}, skipping.")
            continue
        resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path / img_path.name), resized)

    print(f"  Done.")
    return str(out_path)


def apply_clahe(
    input_dir: str = "data/images",
    output_dir: str = "data/images",
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> str:
    """
    Apply CLAHE to each image in input_dir and save results to output_dir.
    CLAHE is applied per channel in LAB color space to avoid shifting hue.

    Args:
        input_dir: Directory containing source images.
        output_dir: Directory to save processed images.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of the grid for histogram equalization.

    Returns:
        Path to the output directory as a string.
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print(f"No images found in: {input_dir}")
        return str(out_path)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    print(f"Applying CLAHE to {len(image_files)} images -> {out_path}")
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path.name}, skipping.")
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        cv2.imwrite(str(out_path / img_path.name), result)

    print(f"  Done.")
    return str(out_path)


def normalize_images(
    input_dir: str = "data/images",
    output_dir: str = "data/images",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> str:
    """
    Normalize images using per-channel mean and std and save as .npy files.
    Defaults to ImageNet mean/std (RGB order), suitable for pretrained models.

    Args:
        input_dir: Directory containing source images.
        output_dir: Directory to save normalized .npy files.
        mean: Per-channel mean in RGB order.
        std: Per-channel std in RGB order.

    Returns:
        Path to the output directory as a string.
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

    if not image_files:
        print(f"No images found in: {input_dir}")
        return str(out_path)

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    print(f"Normalizing {len(image_files)} images -> {out_path}")
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not read {img_path.name}, skipping.")
            continue

        # Convert BGR -> RGB, scale to [0, 1], then standardize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (img_rgb - mean_arr) / std_arr

        np.save(str(out_path / img_path.stem), normalized)

    print(f"  Done.")
    return str(out_path)


if __name__ == "__main__":
    raw_dir = "data/images"
    resized_dir = "data/processed/resized"
    clahe_dir = "data/processed/clahe"
    normalized_dir = "data/processed/normalized"

    resized_dir = resize_images(input_dir=raw_dir, size=(224, 224), output_dir=resized_dir)
    clahe_dir = apply_clahe(input_dir=resized_dir, output_dir=clahe_dir)
    normalize_images(input_dir=clahe_dir, output_dir=normalized_dir)
