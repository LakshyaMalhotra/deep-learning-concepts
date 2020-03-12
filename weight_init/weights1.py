"""
Initial Weights and observing training loss
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import helpers

# how many samples per batch to load
batch_size = 100

# %age to training data to use as validation set
valid_pct = 0.2

# convert the dataset to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=True,
                                   transform=transform, download=True)

test_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data', train=False,
                                  transform=transform, download=True)

# obtain training indices
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(num_train * valid_pct))

train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation samples
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# specify the images classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize some training data
# obtain one batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# converting the images from torch tensors to numpy arrays
# to convert the images from 1x28x28 to 28x28
images = np.squeeze(images.numpy())

# fig = plt.figure(figsize=(20, 4))
# for idx in range(20):
#     ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
# ax.imshow(images[idx], cmap='gray')
# ax.set_title(classes[labels[idx]])

# plt.show()

#####################################################################
# INITIALIZE WEIGHTS
#####################################################################
# define the NN arch


class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=64, constant_weight=None):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)

        # add a dropout layer
        self.dropout = nn.Dropout(0.25)

        # initialize the weights to specified, constant value
        if (constant_weight is not None):
            for m in self.modules():        # iterate through every layer
                if isinstance(m, nn.Linear):
                    # initialize the weight to constants
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # flatten the image
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# compare the model behavior
model_0 = Net(constant_weight=0)
model_1 = Net(constant_weight=1)

# put them in a list to compare
model_list = [(model_0, 'All zeros'),
              (model_1, 'All ones')]

# plot the loss over first 100 batches
# helpers.compare_init_weights(model_list, 'All zeros vs ones',
#                              train_loader, valid_loader)
plt.show()

# Using a uniform distribution
# We can use PyTorch `m.weight.data.uniform_` method where 'm' is model
# to change the weights.
# Let's define a function that takes in a module and applies
# specified weight initialization


def weights_init_uniform(m):
    classname = m.__class__.__name__
    print(classname, type(classname))
    # string method: if match not found it returns -1
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias 0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


model_uniform = Net()
model_uniform.apply(weights_init_uniform)

# evaluate behavior
# helpers.compare_init_weights([(model_uniform, 'Uniform Weights')],
#                              'Uniform Baseline',
#                              train_loader,
#                              valid_loader)

# general rule for setting the weights


def weights_init_uniform_center(m):
    classname = m.__class__.__name__
    # for every layer in the network
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.5, 0.5)
        m.bias.data.fill_(0)


# create a new model with these weights
model_centered = Net()
model_centered.apply(weights_init_uniform_center)

# evaluate behavior
helpers.compare_init_weights([(model_centered, 'Uniform centered weights')],
                             'Uniform Baseline vs. Uniform Centered',
                             train_loader,
                             valid_loader)

# As we can see, uniform centered initilaization performed way better than
# just uniform initialization (or any previously described initialization)
# Let's try the general rule for weight initialization, i.e. [-y,y] where
# y=1/\sqrt(n)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every linear layer in the model
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


model_rule = Net()
model_rule.apply(weights_init_uniform_rule)


model_list = [(model_centered, 'Centered weights [-0.5, 0.5)'),
              (model_rule, 'General rule [-y, y)')]

# evaluate the behavior
helpers.compare_init_weights(model_list,
                             '[-0.5,0.5) vs. [-y, y)',
                             train_loader,
                             valid_loader)


def weights_init_normal_rule(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.normal_(0.0, y)
        m.bias.data.fill_(0)


model_normal_rule = Net()
model_normal_rule.apply(weights_init_normal_rule)

model_list = [(model_centered, 'Centered weights [-0.5, 0.5)'),
              (model_rule, 'General rule [-y, y)'),
              (model_normal_rule, 'General normal rule [-y, y)')]
helpers.compare_init_weights(model_list,
                             'Centered vs Uniform vs Normal',
                             train_loader,
                             valid_loader)

# Instantiate a model with _no_ explicit weight initialization
net = Net()
# evaluate the behavior using helpers.compare_init_weights
model_list = [(model_centered, 'Centered weights [-0.5, 0.5)'),
              (model_rule, 'General rule [-y, y)'),
              (model_normal_rule, 'General normal rule [-y, y)'),
              (net, 'PyTorch default')]
helpers.compare_init_weights(model_list, "Comparison",
                             train_loader, valid_loader)
