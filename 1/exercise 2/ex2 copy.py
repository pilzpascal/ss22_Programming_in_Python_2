def validate_images(input_dir: str, output_dir: str, log_file: str, formatter: str = "06d") -> int:
    import os
    # import glob
    from PIL import Image
    import numpy as np

    max_file_size = 250000
    min_pixel_size = 96
    invalid_files = []
    hashes = []
    file_names = []
    valid_images = 0

    formatter = ''.join(["{:", formatter, "}"])
    # "converting" the formatter into the correct form, for example '06d' would become '{:06d}'

    os.makedirs(os.path.relpath(output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    input_dir = os.path.abspath(input_dir)
    # I am using os.walk instead of glob.glob because with glob.glob i did not find a good way of getting all file types
    # while not getting directories, and this is made easier with os.walk
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # storing tuples of the path and the filename, to keep them separated but not lose the information
            file_names.append((root, file))
    # file_names = [os.path.relpath(file) for file in glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)]
    file_names = sorted(file_names, key=lambda tup: tup[1])
    # Sorting by the second element in the tuples, i.e., the file name

    for number, f in enumerate(file_names):

        if not (os.path.basename(f[1]).endswith(".jpg") or f[1].endswith(".JPG")
                or f[1].endswith(".jpeg") or f[1].endswith(".JPEG")):
            error_code = 1
            invalid_files.append(f"{f[1]},{error_code}\n")
            continue

        if os.path.getsize(os.path.join(f[0], f[1])) > max_file_size:
            error_code = 2
            invalid_files.append(f"{f[1]},{error_code}\n")
            continue

        try:
            image = Image.open(os.path.join(f[0], f[1]), mode="r")
            if image.mode != "RGB" or image.size[0] < min_pixel_size or image.size[1] < min_pixel_size:
                error_code = 4
                invalid_files.append(f"{f[1]},{error_code}\n")
                image.close()
                continue
            im = np.asarray(image)

            if np.var(image.convert('L')) == 0:
                error_code = 5
                invalid_files.append(f"{f[1]},{error_code}\n")
                image.close()
                continue

            image.close()
        except Exception:
            error_code = 3
            invalid_files.append(f"{f[1]},{error_code}\n")
            continue
        else:
            image.close()

        im_hash = hash(im.tostring())
        if im_hash in hashes:
            error_code = 6
            invalid_files.append(f"{f[1]},{error_code}\n")
            continue
        hashes.append(im_hash)
        im = Image.fromarray(im)
        im.save(os.path.relpath(os.path.join(output_dir, formatter.format(number + 1) + ".jpg")))
        valid_images += 1

    with open(log_file, "a") as f:
        for line in invalid_files:
            f.write(line)
    return valid_images
