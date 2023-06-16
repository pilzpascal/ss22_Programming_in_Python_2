import os
import glob
import csv
import utils
import datasets
import architectures
import torch
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ================================================================================
# the following are parameters that could be given as input parameters later

# The paths for the csv files containing the indices of the training, test, val data and the mea and std
mean_and_std_file_path = "mean_and_std.csv"
indices_file_path = "data_indices.csv"
# select device
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# Prepare a path to plot to
results_path = "results"

# training hyperparameters
batch_size = 1
weight_decay = 1e-5
learning_rate = 1e-3
n_updates = 1_000

randomness = False
random_seed = 0

if not randomness:
    utils.set_seed(random_seed)

# Prepare a path to plot to
plotpath = os.path.join(results_path, "plots")
os.makedirs(plotpath, exist_ok=True)

data_path = "/Users/pascal/Library/Mobile Documents/com~apple~CloudDocs/Linz/JKU/AI Bachelor/2. Semester/" \
            "Programming in Python 2/Programming_in_Python_2_Exercise_Assignments/2/data"
file_names = glob.glob(os.path.join(data_path, "**", "*.jpg"), recursive=True)
file_names.sort()

if not os.path.exists(indices_file_path):
    # the hard-coded numbers are bc i want to have val and test sets separated by students
    # and i am too lazy to code a nice solution. This roughly splits it into 2:1
    train_size = 27_430
    test_size = 1_792
    val_size = 188

    # saving which indices correspond to which images
    header = ["index", "set", "folder", "file_name"]
    csv_data = []
    for index, entry in enumerate(file_names):
        mode = "train" if index < train_size else ("test" if index < train_size + test_size else "val")
        csv_data.append([index, mode, os.path.split(os.path.dirname(entry))[1], os.path.split(entry)[1]])
    with open(indices_file_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(csv_data)
else:
    with open(indices_file_path, "r") as csv_file:
        train_size, test_size, val_size = 0, 0, 0
        reader = csv.reader(csv_file)
        header = next(reader)
        for row in reader:
            if row[1] == "train":
                train_size += 1
            elif row[1] == "test":
                test_size += 1
            elif row[1] == "val":
                val_size += 1
            else:
                raise ValueError(f"Encountered unexpected while reading csv file containing the saved indices.\n"
                                 f"Should be in ['train', 'test', 'val'] but is {row[1]}")

train_size = 1
test_size = 1
val_size = 1

data_set = datasets.ImageData(file_names=file_names)

train_set = torch.utils.data.Subset(data_set, indices=range(0, train_size))
test_set = torch.utils.data.Subset(data_set, indices=range(train_size, train_size + test_size))
val_set = torch.utils.data.Subset(data_set, indices=range(train_size + test_size, train_size + test_size + val_size))

print("=" * 80)
print(f"Data was split into training data, test data, "
      f"and validation data with a size\nof {len(train_set)} : {len(test_set)} : {len(val_set)}, "
      f"i.e., a ratio of {(len(train_set) / len(test_set)):.4f} : 1 : {(len(val_set) / len(test_set)):.4f}. "
      f"We have {len(data_set)}\nimages in total.")
print("=" * 80 + "\n")

# getting the mean and standard deviation for the training set
# Since this computation can be more intense we save the values
standardizer = utils.ImageStandardizer(files=file_names[:train_size])
if not os.path.exists(mean_and_std_file_path):
    mean, std = standardizer.analyze_images()
    with open(mean_and_std_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mean", "std"])
        writer.writerows([mean_val, std_val] for mean_val, std_val in zip(mean, std))
else:
    with open(mean_and_std_file_path, "r") as f:
        reader = csv.DictReader(f)
        mean, std = [], []
        for row in reader:
            mean.append(float(row["mean"]))
            std.append(float(row["std"]))

mean = [0, 0, 0]
std = [1, 1, 1]

print("=" * 80)
print(f"We have a mean over the training set of\n{mean}, a standard\ndeviation of {std}")
print("=" * 80 + "\n")

# create datasets and data_loaders for cropped images without any further augmentation
train_set_eval = datasets.TransformedImages(train_set, mean=mean, std=std)
test_set_transformed = datasets.TransformedImages(test_set, mean=mean, std=std)
val_set_transformed = datasets.TransformedImages(val_set, mean=mean, std=std)

train_loader = torch.utils.data.DataLoader(
    train_set_eval,
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    test_set_transformed,
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    val_set_transformed,
    batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# create dataset and data_loader for training images with cropping and augmentation
transform_chain = None
if randomness:
    transform_chain = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomCrop(size=[100, 100])
    ])

train_set_augmented = datasets.TransformedImages(
    train_set, mean=mean, std=std, transform_chain=transform_chain, random=randomness)
train_loader_augmented = torch.utils.data.DataLoader(
    train_set_augmented,
    batch_size=batch_size, shuffle=randomness, num_workers=0, pin_memory=True)

# Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))

# Create Network
net = architectures.CNN1()
net.to(device)

# Get mse loss function
# loss_fn = torch.nn.MSELoss
loss_fn = torch.nn.MSELoss()

# Get adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

print_stats_at = 100  # print status to tensorboard every x updates
plot_at = 1000
validate_at = 300  # evaluate model on validation set and check for new best model every x updates
update = 0  # current update counter
not_saved = 0  # used for early stopping, how often we did not see an improvement
best_validation_loss = np.inf  # best validation loss so far

update_progress_bar = tqdm(total=train_size // batch_size, desc=f"loss: {np.nan:7.5f}", position=0)

# Save initial model as "best" model (will be overwritten later)
saved_model_file = os.path.join(results_path, "best_model.pt")
torch.save(net, saved_model_file)

print("=" * 80)
print("Training begins.")
print("=" * 80 + "\n")

while update < n_updates:
    for data in train_loader_augmented:
        inputs, targets, ids = data
        targets = targets.to(device)
        inputs = inputs.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Get outputs of our network
        outputs = net(inputs)

        # Calculate loss, do backward pass and update weights
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # Print current status and score
        if (update + 1) % print_stats_at == 0:
            writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

        # Plot output
        if (update + 1) % plot_at == 0:
            utils.plot(inputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), outputs.detach().cpu().numpy(),
                       plotpath, update)

        # Evaluate model on validation set
        if (update + 1) % validate_at == 0:
            val_loss = utils.evaluate_model(model=net, data_loader=val_loader, loss_fn=loss_fn, device=device,
                                            mean=mean, std=std)
            writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
            # Add weights and gradients as arrays to tensorboard
            for i, (name, param) in enumerate(net.named_parameters()):
                writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)
                writer.add_histogram(tag=f"validation/gradients_{i} ({name})", values=param.grad.cpu(),
                                     global_step=update)
            # Save best model for early stopping
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                torch.save(net, saved_model_file)

        update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
        update_progress_bar.update()
        update += 1
        if update >= n_updates:
            break

update_progress_bar.close()
writer.close()
print("Finished Training!")

# Load best model and compute score on test set
print(f"Computing scores for best model")
net = torch.load(saved_model_file)
train_loss = utils.evaluate_model(net, data_loader=train_loader, loss_fn=loss_fn, device=device, mean=mean, std=std)
val_loss = utils.evaluate_model(net, data_loader=val_loader, loss_fn=loss_fn, device=device, mean=mean, std=std)
test_loss = utils.evaluate_model(net, data_loader=test_loader, loss_fn=loss_fn, device=device, mean=mean, std=std)

print(f"Scores:")
print(f"  training loss: {train_loss}")
print(f"validation loss: {val_loss}")
print(f"      test loss: {test_loss}")

# Write result to file
with open(os.path.join(results_path, "results.txt"), "w") as rf:
    print(f"Scores:", file=rf)
    print(f"  training loss: {train_loss}", file=rf)
    print(f"validation loss: {val_loss}", file=rf)
    print(f"      test loss: {test_loss}", file=rf)

# batch = next((iter(test_loader)))
# batch = batch[0][0]
# batch = net(batch)
# print(len(batch[0]))
# print([len(elem) for elem in batch])
# print(len(image))
# print(image)
# test_image = batch[0][:][:][:]
# print(test_image[:3].shape)
# print(std.shape, mean.shape)
# test_image = torch.permute(test_image[:3], (1, 2, 0))
# test_image[:] *= std
# test_image[:] += mean
# trans = transforms.ToPILImage()
# test_image = trans(torch.permute(test_image, (2, 0, 1)))
# test_image.show()
