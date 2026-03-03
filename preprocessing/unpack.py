import csv
import zipfile
from pathlib import Path


def unpack_data(file_paths: list[str], output_dir: str = "data/images") -> None:
    """
    Unpack image data zip files from the given file paths.
    
    Args:
        file_paths: List of zip file paths to unpack
        output_dir: Output directory for extracted files. Defaults to 'data/images'.
    """
    for file_path in file_paths:
        zip_path = Path(file_path)
        
        if not zip_path.exists():
            print(f"Warning: File does not exist: {file_path}")
            continue
        
        if not zip_path.suffix == ".zip":
            print(f"Warning: Not a zip file: {file_path}")
            continue
        
        extract_to = Path(output_dir) if output_dir else zip_path.parent
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Unpacking: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    # Skip directories
                    if member.endswith('/'):
                        continue
                    # Remove the top-level folder from the path
                    parts = Path(member).parts
                    if len(parts) > 1:
                        # Strip the first directory (e.g., "sample/img.jpg" -> "img.jpg")
                        target_path = extract_to / Path(*parts[1:])
                    else:
                        target_path = extract_to / member
                    
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Extract the file
                    with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
            print(f"  -> Extracted to: {extract_to}")
        except zipfile.BadZipFile:
            print(f"  -> Error: Invalid zip file: {zip_path}")
        except Exception as e:
            print(f"  -> Error unpacking {zip_path}: {e}")


def unpack_label(file_paths: list[str], output_dir: str = None) -> None:
    """
    Unpack label csv zip files from the given file paths.
    
    Args:
        file_paths: List of zip file paths to unpack
        output_dir: Output directory for extracted files. If None, extracts to same folder as zip.
    """
    for file_path in file_paths:
        zip_path = Path(file_path)
        
        if not zip_path.exists():
            print(f"Warning: File does not exist: {file_path}")
            continue
        
        if not zip_path.suffix == ".zip":
            print(f"Warning: Not a zip file: {file_path}")
            continue
        
        extract_to = Path(output_dir) if output_dir else zip_path.parent
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"Unpacking label: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"  -> Extracted to: {extract_to}")
        except zipfile.BadZipFile:
            print(f"  -> Error: Invalid zip file: {zip_path}")
        except Exception as e:
            print(f"  -> Error unpacking {zip_path}: {e}")


def image_to_score(csv_path: str) -> dict[str, str]:
    """
    Read a CSV file with 'image' and 'level' columns and return a mapping.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        A dict mapping image name to level.
    """
    result = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["image"]] = row["level"]
    return result


if __name__ == "__main__":
    # Example usage
    base_dir = "data"
    
    # Unpack image data to data/images
    unpack_data([
        f"{base_dir}/sample.zip"
    ], output_dir=f"{base_dir}/images")
    
    # Unpack labels
    unpack_label([
        f"{base_dir}/label/trainLabels.csv.zip",
    ])
