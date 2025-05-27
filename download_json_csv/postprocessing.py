def remove_leading_lines_from_csv(input_filename, output_filename, lines_to_remove=5):
    """
    Removes a specified number of leading lines from a single CSV file.
    Also removes surrounding quotes from headers and values.

    Args:
        input_filename (str): The path to the original CSV file.
        output_filename (str): The path to save the modified CSV file.
        lines_to_remove (int): The number of lines to remove from the beginning.
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

        # Remove surrounding quotes from headers and values
        cleaned_lines = []
        for line in lines_to_keep:
            # Split line by comma, strip quotes, and rejoin
            cleaned_line = ','.join(field.strip().strip('"') for field in line.split(','))
            cleaned_lines.append(cleaned_line + '\n')

        with open(output_filename, 'w', newline='', encoding='utf-8') as outfile:
            outfile.writelines(cleaned_lines)

        print(f"Processed '{input_filename}' -> '{output_filename}'")
        return True

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        return False
    except Exception as e:
        print(f"An error occurred while processing '{input_filename}': {e}")
        return False
