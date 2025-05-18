import os
import random
import shutil

from tqdm import tqdm


def split_files(input_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # Define output directories inside the input folder
    train_folder = os.path.join(input_folder, "train")
    val_folder = os.path.join(input_folder, "val")
    test_folder = os.path.join(input_folder, "test")

    # Create the directories if they don't exist
    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(folder, exist_ok=False)

    # List all .npz files in the input folder (ignoring subdirectories)
    files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".npz") and os.path.isfile(os.path.join(input_folder, f))
    ]

    # Shuffle the files for randomness
    random.seed(seed)
    random.shuffle(files)

    total_files = len(files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    # The remaining files will be used for testing
    test_count = total_files - train_count - val_count

    print(f"Total files: {total_files}")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

    # Split the file list
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # Move files to their respective folders
    for filename in tqdm(train_files):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(train_folder, filename)
        shutil.move(src_path, dst_path)

    for filename in tqdm(val_files):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(val_folder, filename)
        shutil.move(src_path, dst_path)

    for filename in tqdm(test_files):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(test_folder, filename)
        shutil.move(src_path, dst_path)

if __name__ == "__main__":
    # Replace with the path to your folder containing .npz files
    input_folder = None
    split_files(input_folder)
