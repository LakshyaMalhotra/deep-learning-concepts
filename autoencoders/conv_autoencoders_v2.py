# import the packages
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST('~/.pytorch/MNIST_data', train=True, download=True,
                            transform=transform)
test_data = datasets.MNIST('~/.pytorch/MNIST_data', train=False, download=True,
                           transform=transform)

# defining the batch-size and creating the dataloaders
bs = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=True)

# Visualize the data
# obtain one batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

print(images.shape)
print(img.shape)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, xticks=[], yticks=[])


# ax.imshow(img, cmap='gray')
# plt.show()

# Convolutional Autoencoder
# define the NN arch

class ConvAutoencoderV2(nn.Module):
    def __init__(self):
        super(ConvAutoencoderV2, self).__init__()

        ## Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        ## Decoder layers
        self.conv3 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        ## Encoder
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        ## Decoder
        # x = F.upsample(x, scale_factor=2, mode='nearest')   --> F.upsample is deprecated
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.conv3(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.conv4(x))

        return x


# initialize the NN
model = ConvAutoencoderV2()
print(model)

# specify the loss and the optimizer
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# number of epochs
n_epochs = 10

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0

    ## train the model
    for images, _ in train_loader:
        # clear the gradients of the optimizer
        optimizer.zero_grad()

        # feed forward
        outputs = model(images)

        # calculate the loss
        loss = criterion(outputs, images)

        # backward pass
        loss.backward()

        # perform a single optimization step
        optimizer.step()

        # update the running training loss
        train_loss += loss.item() * images.size(0)

    # print average training stats
    train_loss /= len(train_loader)
    print("Epoch: {} \t Training loss: {:.6f}".format(epoch, train_loss))
    torch.save(model.state_dict(), 'conv_autoencoder_v2.pt')

# Loading the weights of the trained models
state_dict = torch.load('conv_autoencoder_v2.pt')
model.load_state_dict(state_dict)

# Checking out the results
# Obtain one batch of the test images
dataiter = iter(test_loader)
images, _ = next(dataiter)

# get outputs
with torch.no_grad():
    outputs = model(images)

# prep images for display
images = images.numpy()
print(images.shape)

# output is resized into a batch of images
# outputs = outputs.view(outputs.shape[0], 1, 28, 28)
outputs = outputs.numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex='all', sharey=True, figsize=(20, 5))

# input images on top row, reconstructions on bottom
for images, row in zip([images, outputs], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.suptitle('Convolutional Upsampled Autoencoders ', fontsize=20)
plt.savefig("conv_autoencoder_v2.png", dpi=200)
plt.show()
