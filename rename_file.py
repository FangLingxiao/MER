import os

folder_path = "/home/s5614279/Media Data Analytics and Modelling/MER/data/PMEmo2019/PMEmo2019/chorus"

mp3_files = [file for file in os.listdir(folder_path) if file.endswith('.mp3')]

def sort_by_number(filename):
    return int(''.join(filter(str.isdigit, filename)))

mp3_files.sort(key=sort_by_number)

for i, file in enumerate(mp3_files):
    new_name = f"{i+200}.mp3"
    old_path = os.path.join(folder_path, file)
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)

    print(f"rename {file} as {new_name}")

print("rename successfully!")
