import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and the test datasets
train_data = datasets.MNIST("~/.pytorch/MNIST_data", train=True, download=True,
                            transform=transform)
test_data = datasets.MNIST("~/.pytorch/MNIST_data", train=False, download=True,
                           transform=transform)

# defining the batch size and create the dataloaders
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

# Linear Autoencoder
# Since the images are normalized between 0 and 1, we need to use a sigmoid activation on the
# output layer to get values that match this input value range.
# Define the NN arch
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()

        ## encoder
        self.encoder = nn.Linear(784, encoding_dim)

        ## decoder
        self.decoder = nn.Linear(encoding_dim, 784)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.encoder(x))
        x = torch.sigmoid(self.decoder(x))

        return x


dim: int = 32
model = Autoencoder(dim)
print(model)

# specify the loss function
criterion = nn.MSELoss()

# specify the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# number of epochs
n_epochs = 20

# for epoch in range(1, n_epochs+1):
#     # monitor training loss
#     train_loss = 0.0
#
#     ## train the model
#     for images, _ in train_loader:
#         # flatten images
#         images = images.view(images.shape[0], -1)
#
#         # clear the gradients of the optimizer
#         optimizer.zero_grad()
#
#         # feed forward
#         outputs = model(images)
#
#         # calculate the loss
#         loss = criterion(outputs, images)
#
#         # backward pass
#         loss.backward()
#
#         # perform a single optimization step
#         optimizer.step()
#
#         # update the running training loss
#         train_loss += loss.item() * images.size(0)
#
#     # print average training stats
#     train_loss /= len(train_loader)
#     print("Epoch: {} \t Training loss: {:.6f}".format(epoch, train_loss))
#     torch.save(model.state_dict(), 'autoencoder.pt')

# Loading the weights of the trained models
state_dict = torch.load('autoencoder.pt')
model.load_state_dict(state_dict)

# Checking out the results
# Obtain one batch of the test images
dataiter = iter(test_loader)
images, _ = next(dataiter)

img_flatten = images.view(images.shape[0], -1)

# get outputs
with torch.no_grad():
    output = model(img_flatten)

# prep images for display
images = images.numpy()
print(images.shape)

# output is resized into a batch of images
output = output.view(output.shape[0], 1, 28, 28)
output = output.numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 5))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# fig.suptitle('Linear Autoencoders', fontsize=20)
# plt.savefig("linear_autoencoder.png", dpi=200)
plt.show()