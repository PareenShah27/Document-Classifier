import os
import glob
import re
from typing import Tuple, List, Any

# Try importing PDF Library
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

def clean_text(text: Any) -> str:
    """
    Applies text cleaning: lowercase, remove special chars/numbers.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+',' ', text)
    return text

def read_doc(file_path: str) -> str | None:
    """
    Reads text from a file (TXT or PDF).
    Returns the raw string content.
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    content = ""

    try:
        # Read .txt files
        if ext == ".txt":
            with open(file_path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
                content = f.read()
        
        # Read .pdf files
        elif ext == ".pdf":
            if not PDF_SUPPORT:
                print("Error: PDF detected but 'pypdf' is not installed.")
                print("Run: pip install pypdf")
                return None
            
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + " "
        
        # For all other file types
        else:
            print(f"Error: Unsupported file format '{ext}'. Use .txt or .pdf")
            return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    return content

def load_data(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Loads data from sports/ and politics/ directories.
    """
    categories = ['sports', 'politics']
    cleaned_text = []
    labels = []

    print(f"Loading data from {data_dir}")
    for category in categories:
        path = os.path.join(data_dir, category)
        files = glob.glob(os.path.join(path, "*.txt")) # Extract .txt file paths
        files.extend(glob.glob(os.path.join(path, "*.pdf"))) # Extract and insert .pdf file paths

        # Extract data from files
        for file in files:
            content = read_doc(file)
            if content:
                cleaned_text.append(clean_text(content))
                labels.append(category)

    return cleaned_text, labels