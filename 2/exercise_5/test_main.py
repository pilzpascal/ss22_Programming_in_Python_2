import os
import glob
import csv
import utils
import datasets
import torch
import torch.utils.data
from torchvision import transforms
import numpy as np

np.random.seed(0)
torch.manual_seed(0)

data_path = "/Users/pascal/Library/Mobile Documents/com~apple~CloudDocs/Linz/JKU/AI Bachelor/2. Semester/" \
            "Programming in Python 2/Programming_in_Python_2_Exercise_Assignments/2/data"
file_names = glob.glob(os.path.join(data_path, "**", "*.jpg"), recursive=True)
file_names.sort()

# the hard-coded numbers are bc i want to have val and test sets separated by students
# and i am too lazy to code a nice solution. This roughly splits it into 2:1
train_size = 27_430
test_size = 1_792
val_size = 188

# saving which indices correspond to which images
header = ["index", "set", "folder", "file_name"]
csv_data = []
for index, entry in enumerate(file_names):  # doing this explicitly instead of in a list comprehension for readability
    mode = "train" if index < train_size else ("test" if index < train_size + test_size else "val")
    csv_data.append([index, mode, os.path.split(os.path.dirname(entry))[1], os.path.split(entry)[1]])
with open("data_indices.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)
    writer.writerows(csv_data)

data_set = datasets.ImageData(file_names=file_names)

train_set = torch.utils.data.Subset(data_set, indices=range(0, train_size))
test_set = torch.utils.data.Subset(data_set, indices=range(train_size, train_size + test_size))
val_set = torch.utils.data.Subset(data_set, indices=range(train_size + test_size, len(file_names)))

print("="*80)
print(f"Data was split into training data, test data, "
      f"and validation data with a ratio\nof {len(train_set)} : {len(test_set)} : {len(val_set)}, "
      f"i.e., a ratio of {(len(train_set)/len(test_set)):.4f} : 1 : {(len(val_set)/len(test_set)):.4f}. "
      f"We have {len(data_set)}\nimages in total.")
print("="*80 + "\n")

# getting the mean and standard deviation for the training set
# Since this computation can be more intense we save the values
standardizer = utils.ImageStandardizer(files=file_names[:train_size])
mean_and_std_file_path = "mean_and_std.csv"
if not os.path.exists(mean_and_std_file_path):
    mean, std = standardizer.analyze_images()
    with open(mean_and_std_file_path, "w") as f:
        writer = csv.DictWriter(f, ["mean", "std"])
        writer.writeheader()
        writer.writerow({"mean": mean, "std": std})
else:
    with open(mean_and_std_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mean = row["mean"].astype(np.float64)
            std = row["std"].astype(np.float64)

print(mean, std)
