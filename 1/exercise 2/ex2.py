"""
Author: Pascal Pilz
Matr.Nr.: K12111234
Exercise 2
"""
import os
from PIL import Image
import numpy as np
import shutil


def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "06d") -> int:
    max_file_size = 250000
    min_pixel_size = 96
    invalid_files = []
    hashes = []
    file_names = []
    valid_images = 0

    formatter = ''.join(["{:", formatter, "}"])
    # Converting the formatter into the correct form; for example '06d' becomes '{:06d}'

    os.makedirs(os.path.relpath(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    input_dir = os.path.abspath(input_dir)
    # Here we use os.walk instead of glob.glob because with glob.glob i did not find a good way of getting all file
    # types while not getting directories, and this is made easier with os.walk
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Storing tuples of the path and the filename in order to keep them separated, which we need for sorting
            file_names.append(os.path.join(root, file))
    file_names.sort()

    for f in file_names:
        # To avoid checking .DS_Store files we implement this condition
        #if os.path.relpath(f, start=input_dir).startswith("."):
            #continue

        # Checking Rule 1: files must end with .jpg, .JPG, .jpeg, .JPEG
        if not (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".jpeg") or f.endswith(".JPEG")):
            error_code = 1
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue

        # Checking Rule 2: files must be <= 250000 bits
        if os.path.getsize(f) > max_file_size:
            error_code = 2
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue

        # Opening the image in a try-except block, such that i can catch the exception if Image.open fails and thus
        # Checking Rule 3: files must be readable as images
        try:
            image = Image.open(f, mode="r")
            # Checking Rule 4: images must be of size (H,W,3), with H,W >= 96
            if image.mode != "RGB" or image.size[0] < min_pixel_size or image.size[1] < min_pixel_size:
                error_code = 4
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
                image.close()
                continue
            image_as_array = np.array(image)

            # Checking Rule 5: image variance > 0, i.e., images cannot be just one color
            # I am doing this by converting them into grayscale and just taking the variance over the result
            if np.var(image.convert('L')) == 0:
                error_code = 5
                invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
                image.close()
                continue

            image.close()
        except Exception:
            # We have to catch "Except", as the exception that is raised if Image.open fails does not inherit from
            # BaseException
            error_code = 3
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue
        else:
            image.close()

        # Hashing the images; converting them to string makes it immutable and thus hashable
        im_hash = hash(image_as_array.tostring())
        # Checking Rule 6: no images shall be copied more than once
        if im_hash in hashes:
            error_code = 6
            invalid_files.append(f"{os.path.relpath(f, start=input_dir)};{error_code}\n")
            continue
        hashes.append(im_hash)

        # Preparing the file name/path
        file_name = os.path.relpath(os.path.join(output_dir, formatter.format(valid_images) + ".jpg"))
        # We copy the images instead of saving them manually in order to avoid compression artefacts and the like
        shutil.copy(f, file_name)
        # This counter is used for the return statement, but also for numbering the checked and copied files
        valid_images += 1

    # Writing the log file
    with open(log_file, "a") as f:
        for line in invalid_files:
            f.write(line)
    return valid_images
