import os
import sys

def list_directory_contents(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            print(f"Contents of directory '{path}':")
            for item in os.listdir(path):
                print(f" - {item}")
        elif os.path.isfile(path):
            print(f"The provided path is a file: '{path}'")
        else:
            print(f"The path '{path}' is neither a file nor a directory.")
    else:
        print(f"The path '{path}' does not exist.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
        list_directory_contents(directory_path)
    else:
        print("Please provide a directory or file path as an argument.")
