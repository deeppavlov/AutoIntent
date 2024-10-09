import os


def list_and_replace_colon_in_filenames(directory: str) -> None:
    """
    List all files in the specified directory and its subdirectories.
    If a filename contains ':', replace it with '_'.

    Args:
        directory (str): The path of the directory to scan.

    Returns:
        None
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if ':' in filename:
                new_filename = filename.replace(':', '_')
                old_file_path = os.path.join(root, filename)
                new_file_path = os.path.join(root, new_filename)
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {old_file_path} -> {new_file_path}")

        for dirname in dirs:
            if ':' in dirname:
                new_dirname = dirname.replace(':', '_')
                old_dir_path = os.path.join(root, dirname)
                new_dir_path = os.path.join(root, new_dirname)
                os.rename(old_dir_path, new_dir_path)
                print(f"Renamed directory: {old_dir_path} -> {new_dir_path}")


if __name__ == "__main__":
    list_and_replace_colon_in_filenames(".")
