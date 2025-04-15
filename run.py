import os
from USBTypeDetector import USBTypeDetector, Config

if __name__ == "__main__":

    root_dir = "data/classes_greyscale_augmented"
    save_dir = "data/classes_greyscale_preprocessed"

    os.makedirs(save_dir, exist_ok=True)
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        
        if os.path.isdir(subdir_path):
            # print(f"Images in {subdir}:")
            
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                
                if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"========== Processing {filename} in {subdir} ==========")
                    save_path = os.path.join(save_dir, subdir, filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    USBTypeDetector.run(file_path, save_path)
                    print(f"Saved preprocessed image to {save_path}")
            print()