# src/utils.py

import json
import os

def load_json(filepath):
    """
    Loads a JSON file from the given filepath.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def get_output_path(output_dir, filename):
    """
    Constructs the full path for an output file.

    Args:
        output_dir (str): Directory where the file will be saved.
        filename (str): Name of the file.

    Returns:
        str: Full file path.
    """
    return os.path.join(output_dir, filename)
