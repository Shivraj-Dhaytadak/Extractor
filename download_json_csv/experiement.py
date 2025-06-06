import os

def list_json_files(directory: str, full_path: bool = True) -> list:
    """
    List all JSON files in the given directory.

    Args:
        directory (str): Path to the directory.
        full_path (bool): If True, return full file paths. Else, just filenames.

    Returns:
        List[str]: List of JSON file paths or names.
    """
    files = [
        f if full_path else f
        for f in os.listdir(directory)
        if f.endswith(".json") and os.path.isfile(os.path.join(directory, f))
    ]
    return files

# Example usage:
if __name__ == "__main__":
    json_files = list_json_files("download_json_csv/parsed_output", full_path=True)
    print("Found JSON files:")
    for file in json_files:
        print(" -", file)
