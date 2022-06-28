import os
import glob
import pickle
from torchvision import transforms
import numpy as np

train_path = "/Users/pascal/Library/Mobile Documents/com~apple~CloudDocs/Linz/JKU/AI Bachelor/2. Semester/" \
             "Programming in Python 2/Programming_in_Python_2_Exercise_Assignments/2/train_images"
test_path = "/Users/pascal/Library/Mobile Documents/com~apple~CloudDocs/Linz/JKU/AI Bachelor/2. Semester/" \
            "Programming in Python 2/Programming_in_Python_2_Exercise_Assignments/2/test_images"
val_path = "/Users/pascal/Library/Mobile Documents/com~apple~CloudDocs/Linz/JKU/AI Bachelor/2. Semester/" \
            "Programming in Python 2/Programming_in_Python_2_Exercise_Assignments/2/val_images"

train_file_names = glob.glob(os.path.join(train_path, "**", "*.jpg"), recursive=True)
test_file_names = glob.glob(os.path.join(test_path, "**", "*.jpg"), recursive=True)
val_file_names = glob.glob(os.path.join(val_path, "**", "*.jpg"), recursive=True)

train_file_names.sort()
test_file_names.sort()
val_file_names.sort()

print(len(train_file_names))
print(len(test_file_names))
print(len(val_file_names))
print(train_file_names[0])
