import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

def restructure_val_set(tiny_imagenet_path):
    """
    Restructures the validation directory of Tiny ImageNet to match ImageFolder expected format.
    
    The original structure has all validation images in a single 'images' folder with class
    information in val_annotations.txt. This function reorganizes it to have one subdirectory
    per class, matching how the training data is organized.
    
    Args:
        tiny_imagenet_path (str): Path to the Tiny ImageNet directory.
    """
    val_dir = os.path.join(tiny_imagenet_path, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    
    if not os.path.exists(val_images_dir):
        print("Validation structure already fixed or not in expected format")
        return
    
    # Path to the annotations file
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # Read class labels for validation images
    with open(val_annotations_file, 'r') as f:
        val_annotations = f.readlines()
    
    # Create class directories in val directory
    image_to_class = {}
    for line in val_annotations:
        parts = line.strip().split('\t')
        image_name, class_id = parts[0], parts[1]
        image_to_class[image_name] = class_id
        
        # Create class directory if it doesn't exist
        class_dir = os.path.join(val_dir, class_id)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Move images to their respective class directories
    print("Restructuring validation set...")
    for image_name, class_id in tqdm(image_to_class.items()):
        src_path = os.path.join(val_images_dir, image_name)
        dst_path = os.path.join(val_dir, class_id, image_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
    
    # Remove the now-empty images directory
    if os.path.exists(val_images_dir) and not os.listdir(val_images_dir):
        os.rmdir(val_images_dir)
    
    print("Validation set restructured successfully!")


def download_and_extract_tiny_imagenet(url='http://cs231n.stanford.edu/tiny-imagenet-200.zip', dest_path=None):
    """
    Downloads, extracts, and restructures the Tiny ImageNet dataset.
    
    Args:
        url (str): URL of the Tiny ImageNet zip file.
        dest_path (str): Directory to save the downloaded file and extract its contents.
               If None, uses the data directory in the project root.
    
    Returns:
        str: Path to the extracted Tiny ImageNet directory.
    """
    # If no destination path provided, use the data directory in the project root
    if dest_path is None:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest_path = os.path.join(script_dir, 'data')
        
    # Ensure the destination directory exists
    os.makedirs(dest_path, exist_ok=True)
    
    zip_filename = os.path.join(dest_path, 'tiny-imagenet-200.zip')
    extract_path = os.path.join(dest_path, 'tiny-imagenet-200')
    
    if os.path.exists(extract_path):
        print("Tiny ImageNet is already downloaded and extracted at:", extract_path)
        # Check if we need to restructure the validation set
        val_images_dir = os.path.join(extract_path, 'val', 'images')
        if os.path.exists(val_images_dir):
            print("Restructuring validation set...")
            restructure_val_set(extract_path)
        return extract_path
    
    # Download the dataset if the zip file doesn't exist
    if not os.path.exists(zip_filename):
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_filename)
        print("Download complete.")
    else:
        print("Zip file already exists:", zip_filename)
    
    # Extract the downloaded zip file
    print("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    print("Extraction complete.")
    
    # Restructure the validation set to match the training set format
    restructure_val_set(extract_path)
    
    # Optionally, remove the zip file after extraction to save space
    os.remove(zip_filename)
    print("Removed zip file:", zip_filename)
    
    return extract_path

if __name__ == "__main__":
    dataset_path = download_and_extract_tiny_imagenet()
    print("Tiny ImageNet is ready at:", dataset_path)
