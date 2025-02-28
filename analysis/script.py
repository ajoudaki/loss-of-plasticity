import os
import nbformat
import argparse

def extract_notebook_content(nb_path):
    """
    Extracts code and markdown content from a Jupyter Notebook.
    Returns a list of dictionaries with keys "type" and "source".
    """
    nb = nbformat.read(nb_path, as_version=4)
    cells = []
    for cell in nb.cells:
        if cell.cell_type in ["code", "markdown"]:
            cells.append({
                "type": cell.cell_type,
                "source": cell.source
            })
    return cells

def process_file(file_path, ext):
    """
    Processes a single file based on its extension.
    For Jupyter notebooks (.ipynb), it extracts and prints code & markdown cells.
    For other files, it outputs their content directly.
    """
    print(f"--- Processing: {os.path.basename(file_path)} ---")
    
    if ext == ".ipynb":
        cells = extract_notebook_content(file_path)
        for idx, cell in enumerate(cells, start=1):
            print(f"\nCell {idx} ({cell['type']}):")
            print(cell['source'])
            print("-" * 40)
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    print("\n" + "=" * 80 + "\n")

def process_directory(directory, allowed_extensions):
    """
    Processes all files in the specified directory that match the allowed extensions.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext in allowed_extensions:
            process_file(file_path, ext)

def main():
    parser = argparse.ArgumentParser(
        description="Extract contents from Jupyter notebooks and other files with specified extensions."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan for files (default: current directory)."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".ipynb",
        help=("Comma-separated list of file extensions to process. " 
              "For example: .ipynb,.py,.txt (default: .ipynb)")
    )
    
    args = parser.parse_args()
    # Create a set of allowed extensions, ensuring they are in lowercase and start with a dot.
    allowed_extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    
    process_directory(args.directory, allowed_extensions)

if __name__ == "__main__":
    main()
