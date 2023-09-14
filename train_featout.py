import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import torchvision.transforms as transforms

from models.food101_model import Net
from featout.featout_dataset import Featout
from featout.utils.blur import blur_around_max, zero_out
from featout.interpret import simple_gradient_saliency


# method how to remove features - here by default blurring
BLUR_METHOD = blur_around_max
# BLUR_METHOD = zero_out
# algorithm to derive the model's attention
ATTENTION_ALGORITHM = simple_gradient_saliency
# set this path to some folder, e.g., "outputs", if you want to plot the results
PLOTTING_PATH = None
if PLOTTING_PATH is not None:
    os.makedirs(PLOTTING_PATH, exist_ok=True)

# Point this path to your food-101 dataset directory
FOOD101_PATH = "food-101/images"

# The classes you're interested in
target_classes = ["greek_salad", "beet_salad"]
# Initialize an empty list to collect the indices of target classes
target_indices = []

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the ImageFolder dataset
full_dataset = ImageFolder(FOOD101_PATH, transform=transform)

target_indices = []

# Collect the indices of samples belonging to target classes
for idx, (path, class_idx) in enumerate(full_dataset.imgs):
    if full_dataset.classes[class_idx] in target_classes:
        target_indices.append(idx)

# Create a Subset of the dataset only containing the target classes
target_dataset = Subset(full_dataset, target_indices)

# Create a mapping from old class index to new class index (0 and 1)
class_mapping = {full_dataset.class_to_idx[original_class]: new_class for new_class, original_class in enumerate(target_classes)}

# Modify the labels in target_dataset
for idx in range(len(target_dataset)):
    path, class_idx = target_dataset.dataset.imgs[target_dataset.indices[idx]]
    new_class_idx = class_mapping[class_idx]
    target_dataset.dataset.imgs[target_dataset.indices[idx]] = (path, new_class_idx)

# Use train_test_split to get train and test indices
train_idx, test_idx = train_test_split(target_indices, test_size=0.2, shuffle=True)

# Create train and test Subsets
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

# Wrap the train dataset with Featout
train_dataset = Featout(train_dataset, PLOTTING_PATH)

# Update your DataLoader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# define model and optimizer (standard mnist model from torch is used)
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9
)

for epoch in range(10):
    tic = time.time()
    running_loss = 0.0
    blurred_set = []

    # START FEATOUT
    # first epoch uses unmodified dataset, then we do it every epoch
    # code could be changed to do it only every second epoch or so
    if epoch > 0:
        trainloader.dataset.start_featout(
            net,
            blur_method=BLUR_METHOD,
            algorithm=ATTENTION_ALGORITHM,
        )

    for i, data in enumerate(trainloader):
        # get the inputs
        inputs, labels = data
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (
            i % 200 == 199
        ):  # print every 2000 mini-batches
            print(
                "Epoch %d, samples %5d] loss: %.3f"
                % (epoch + 1, i + 1, running_loss / 2000)
            )
            running_loss = 0.0

    print(f"time for epoch: {time.time()-tic}")

    # Evaluate test performance
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(
        "Accuracy of the network on the test images: %d %%"
        % (100 * correct / total)
    )

    # stop featout after every epoch
    trainloader.dataset.stop_featout()

# Save model
print("Finished Training")
torch.save(
    net.state_dict(), "trained_models/cifar_torchvision.pt"
)
