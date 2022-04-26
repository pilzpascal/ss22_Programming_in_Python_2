"""
Author: Gernot Kucera
Matr.Nr.: k06026191
Exercise ex2
"""

import os
import numpy as np
import shutil
from PIL import Image

def validate_images(input_dir:str, output_dir:str, log_file:str, formatter:str = "06d") -> int:
    # init
    valid_images = 0
    file_names = []
    invalid_files = []
    hashes = []
    # size
    max_file_size = 250000
    min_pixel_size = 96

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # '06d' -> '{:06d}'
    formatter = ''.join(["{:", formatter, "}"])

    input_dir = os.path.abspath(input_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_names.append(os.path.join(root, file))

    file_names.sort()
    #checking rules
    for f in file_names:

        # rule 1
        if not (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".jpeg") or f.endswith(".JPEG")):
            error_code = 1
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
            continue
        # rule 2
        if os.path.getsize(f) > max_file_size:
            error_code = 2
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
            continue
        # rule 3, 4 and 5: try (test images)
        try:
            image = Image.open(f, mode="r")
            if image.mode != "RGB" or image.size[0] < min_pixel_size or image.size[1] < min_pixel_size:
                error_code = 4
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
                image.close()
                continue
            image_as_array = np.array(image)
            if np.var(image.convert('L')) == 0:
                error_code = 5
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
                image.close()
                continue
            image.close()
        except Exception:
            error_code = 3
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
            continue
        else:
            image.close()

        # hash
        im_hash = hash(image_as_array.tostring())
        # rule 6
        if im_hash in hashes:
            error_code = 6
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)}; {error_code}\n")
            continue
        hashes.append(im_hash)

        file_name = os.path.relpath(os.path.join(output_dir, formatter.format(valid_images) + ".jpg"))
        shutil.copy(f, file_name)
        valid_images += 1

        #log file
        with open(log_file, "a") as f:
            for line in invalid_files:
                f.write(line)
    return valid_images



