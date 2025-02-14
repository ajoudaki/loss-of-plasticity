import os
import urllib.request
import zipfile

def download_and_extract_tiny_imagenet(url='http://cs231n.stanford.edu/tiny-imagenet-200.zip', dest_path='.'):
    """
    Downloads and extracts the Tiny ImageNet dataset.
    
    Args:
        url (str): URL of the Tiny ImageNet zip file.
        dest_path (str): Directory to save the downloaded file and extract its contents.
    
    Returns:
        str: Path to the extracted Tiny ImageNet directory.
    """
    zip_filename = os.path.join(dest_path, 'tiny-imagenet-200.zip')
    extract_path = os.path.join(dest_path, 'tiny-imagenet-200')
    
    if os.path.exists(extract_path):
        print("Tiny ImageNet is already downloaded and extracted at:", extract_path)
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
    
    # Optionally, remove the zip file after extraction to save space
    os.remove(zip_filename)
    print("Removed zip file:", zip_filename)
    
    return extract_path

if __name__ == "__main__":
    dataset_path = download_and_extract_tiny_imagenet()
    print("Tiny ImageNet is ready at:", dataset_path)
