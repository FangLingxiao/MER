import os



def rename_mp3_files(folder_path, idx):
    
    """Rename MP3 files in the given folder path with sequential numbers starting from 200.

    Args:
        folder_path (str): The path to the folder containing MP3 files.
    """
        
    mp3_files = [file for file in os.listdir(folder_path) if file.endswith('.mp3')]

    def sort_by_number(filename):
        return int(''.join(filter(str.isdigit, filename)))

    mp3_files.sort(key=sort_by_number)

    for i, file in enumerate(mp3_files):
        new_name = f"{i+idx}.mp3"
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"rename {file} as {new_name}")

    print("rename successfully!")

def calculate_thresholds(data, column, n_levels=3):
    min_value = data[column].min()
    max_value = data[column].max()
    
    thresholds = []
    for i in range(1, n_levels):
        threshold = min_value + i * (max_value - min_value) / n_levels
        thresholds.append(threshold)
    
    return thresholds
