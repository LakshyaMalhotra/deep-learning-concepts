"""
###############################################################################
#    Keys points to be noted from this script:
#    --> `ax.imshow()` plots the image of any matrix
#    --> `fig.add_subplot()` and `ax.annotate` usage 
#    --> How to input user defined weights in the Convolution layer in PyTorch
#    --> How to Visualize the intermediate layers
###############################################################################
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# image path
img_path = 'images/curved_lane.jpg'

# load image
bgr_img = cv2.imread(img_path)

# convert the image to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize the image, rescale entries to lie in [0,1]
gray_img = gray_img.astype(np.float32) / 255.
#  plot the image
# plt.imshow(gray_img, cmap='gray')
# plt.show()

# define the filter (looks like this one is gonna detect vertical edges)
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

print("filter shape: ", filter_vals.shape)

# defining four filters
filter1 = filter_vals
filter2 = -filter_vals
filter3 = filter_vals.T
filter4 = -filter_vals.T

filters = np.array([filter1, filter2, filter3, filter4])

print(filter4)

# Visualize all four filters
# fig = plt.figure(figsize=(10, 5))
# for i in range(4):
#     ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
#     ax.imshow(filters[i], cmap='gray')
#     ax.set_title("Filter %s" %str(i+1))
#     width, height = filters[i].shape
#     for x in range(width):
#         for y in range(height):
#             ax.annotate(str(filters[i][x][y]), xy=(y,x),
#                         horizontalalignment='center',
#                         verticalalignment='center',
#                         color='white' if filters[i][x][y]<0 else 'black')
# plt.show()    

# define a conv layer
class Net(nn.Module):
    def __init__(self, weight):
        super(Net, self).__init__()

        k_height, k_width = weight.shape[2:]

        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)

    def forward(self, x):
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)

        # returns both layers
        return conv_x, activated_x

# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)     # unsqueeze adds one more dimension
print(weight.shape)             # the dimension is (4,1,4,4), argument of unsqueeze takes the dimension
model = Net(weight)

# printing out the model
print(model)

# Visualize the output; default number of filters=4
def viz_layer(layer, n_filters=4):
    fig = plt.figure(figsize=(12, 4))
     
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])

        # grab layer output
        ax.imshow(np.squeeze(layer[0, i].data.numpy()), cmap='gray')
        ax.set_title("Output %s" %str(i+1))

# plot original image
# plt.imshow(gray_img, cmap='gray')

# # visualize all filters
# fig = plt.figure(figsize=(12, 6))
# fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
# for i in range(4):
#     ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
#     ax.imshow(filters[i], cmap='gray')
#     ax.set_title('Filter %s' % str(i+1))

# convert the image into an input tensor
print("Gray image shape", gray_img.shape)
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)
print("Gray image tensor shape", gray_img_tensor.shape)
# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

print("Conv layer shape", conv_layer.shape)
print("Activated layer shape", activated_layer.shape)

conv_layer_squeezed = np.squeeze(conv_layer[0].data.numpy())        # reducing the dimensions from 4D to 3D
print("Squeezed conv layer shape", conv_layer_squeezed.shape)

conv_layer_squeezed1 = np.squeeze(conv_layer[0, 0].data.numpy())        # reducing the dimensions from 4D to 2D
print("Squeezed1 conv layer shape", conv_layer_squeezed1.shape)

conv_layer_squeezed2 = np.squeeze(conv_layer[0, 1].data.numpy())           # reducing the dimension from 4D to 2D for second filter
print("Squeezed2 conv layer shape", conv_layer_squeezed2.shape)

print(np.array_equal(conv_layer_squeezed1, conv_layer_squeezed2))
# vizualize the output
# viz_layer(conv_layer)
# viz_layer(activated_layer)
# plt.show()