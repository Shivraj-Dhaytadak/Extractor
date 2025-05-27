import csv
import os

def remove_leading_lines_from_csv(input_filename, output_filename, lines_to_remove=5):
    """
    Removes a specified number of leading lines from a single CSV file.

    Args:
        input_filename (str): The path to the original CSV file.
        output_filename (str): The path to save the modified CSV file.
        lines_to_remove (int): The number of lines to remove from the beginning.
                               Defaults to 5.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        with open(input_filename, 'r', newline='', encoding='utf-8') as infile:
            all_lines = infile.readlines()

        if len(all_lines) < lines_to_remove:
            print(f"Info: File '{input_filename}' has < {lines_to_remove} lines. Copying as is.")
            lines_to_keep = all_lines
        else:
            lines_to_keep = all_lines[lines_to_remove:]

        cleaned_lines = []
        for line in lines_to_keep:
            # Split line by comma, strip quotes, and rejoin
            cleaned_line = ','.join(field.strip().strip('"') for field in line.split(','))
            cleaned_lines.append(cleaned_line + '\n')
        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            outfile.writelines(cleaned_lines)

        print(f"Processed '{input_filename}' -> '{output_filename}'") # Uncomment for more detail
        return True

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return False
    except Exception as e:
        print(f"An error occurred while processing '{input_filename}': {e}")
        return False

def process_csvs_in_directory(input_dir, output_dir, lines_to_remove=5):
    """
    Processes all CSV files in a given directory, removing leading lines.

    Args:
        input_dir (str): The path to the directory containing input CSV files.
        output_dir (str): The path to the directory where output files will be saved.
        lines_to_remove (int): The number of lines to remove. Defaults to 5.
    """
    # 1. Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # 2. Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: '{output_dir}'")
        except OSError as e:
            print(f"Error creating output directory '{output_dir}': {e}")
            return

    print(f"Scanning '{input_dir}' for CSV files...")
    processed_count = 0
    error_count = 0

    # 3. Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # 4. Check if the file is a CSV file (case-insensitive)
        if filename.lower().endswith('.csv'):
            input_filepath = os.path.join(input_dir, filename)

            # 5. Ensure it's actually a file, not a directory
            if os.path.isfile(input_filepath):
                output_filepath = os.path.join(output_dir, filename)

                # 6. Process the file
                print(f"Processing: {filename}...")
                if remove_leading_lines_from_csv(input_filepath, output_filepath, lines_to_remove):
                   processed_count += 1
                else:
                   error_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count} files.")
    if error_count > 0:
        print(f"Encountered errors with: {error_count} files.")
    print(f"Output files are in: '{output_dir}'")


# --- How to Use ---

# 1. Set the path to the directory containing your input CSV files.
#    Use '.' if the CSV files are in the same directory as the script.
input_directory = r'C:\VScodeMaster\Inferencing\Extractor\download_json_csv\csv'

# 2. Set the path where you want to save the processed files.
#    A new folder name is recommended.
output_directory = r'C:\VScodeMaster\Inferencing\Extractor\download_json_csv\updated_csv'

# 3. Call the function
process_csvs_in_directory(input_directory, output_directory)

# --- Optional: If you want to remove a different number of lines ---
# process_csvs_in_directory(input_directory, output_directory, lines_to_remove=3)