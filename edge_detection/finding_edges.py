"""
###############################################################################
#    Keys points to be noted from this script:
#    --> `ax.imshow()` plots the image of any matrix
#    --> using OpenCV to convert the color of the image 
#    --> using OpenCV to apply a 2D filter to a grayscale image 
#    --> `fig.add_subplot()` and `matplotlib.image` module usage 
#    --> How to Visualize the intermediate layers
###############################################################################
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')

# plt.imshow(image, cmap='gray')
# plt.show()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(gray, cmap='gray')
# plt.show()

# Creating a Sobel kernel for edge detection
sobel_y = np.array([[-1, -2, -1],           # sobel_y is used to detect horizontal edges,
                    [0, 0, 0],              # it takes derivatives along y-axis, hence "sobel_y"
                    [1, 2, 1]])

sobel_x = np.array([[-1, 0, 1],             # sobel_x is used to detect horizontal edges,
                    [-2, 0, 2],             # it takes the derivatives along x-axis. hence "sobel_x"
                    [-1, 0, 1]])  

filter1 = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])

filter2 = np.array([[-1, 0, -1],
                    [0, 4, 0],
                    [-1, 0, -1]])

decimal_filter = np.array([[-0.5, 0, -0.5],
                    [0, 2, 0],
                    [-0.5, 0, -0.5]])

filter_5x5_1 = np.array([[0, -1, -1, -1, 0],
                    [0, 0, -1, 0, 0],
                    [0, 0, 8, 0, 0],
                    [0, 0, -1, 0, 0],
                    [0, -1, -1, -1, 0]])

filter_5x5_2 = np.array([[0, -0.5, -0.5, -0.5, 0],
                    [0, 0, -0.5, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, -0.5, 0, 0],
                    [0, -0.5, -0.5, -0.5, 0]])

def gray_img(img):
    """
    Converts the image to grayscale image.
    """
    img = mpimg.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray

def detect(img, fltr):
    """
    Returns the image convoluted with a given kernel.
    """
    filtered_image = cv2.filter2D(img, -1, fltr)
    return filtered_image

# gray = gray_img('images/curved_lane.jpg')

# filtered_image = detect(gray, sobel_x)

filters = {'sobel_y': sobel_y, 'sobel_x': sobel_x, 'filter_1': filter1,
            'filter_2': filter2, 'decimal_filter': decimal_filter, 
            'filter_5x5_1': filter_5x5_1, 'filter_5x5_2': filter_5x5_2}

imgs = ['curved_lane.jpg', 'bridge_trees_example.jpg', 'white_lines.jpg']


for j in range(len(imgs)):
    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(hspace=0.8)            # add horizontal and vertical space buffer
    for i, (name, fltr) in zip(range(8), filters.items()):
        ax = fig.add_subplot(2, 4, i+1)
        gray = gray_img('images/'+imgs[j])
        # if i != 0:
        gray = detect(gray, fltr)
        ax.set_title(name)
        ax.axis('off')
        ax.imshow(gray, cmap='gray')
    ax = fig.add_subplot(2,4,8)
    gray = gray_img('images/'+imgs[j])
    ax.set_title('gray image')
    ax.axis('off')
    ax.imshow(gray, cmap='gray')
    fig.suptitle(imgs[j], fontsize=16, y=0.98)

    # plt.imshow(filtered_image, cmap='gray')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # plt.savefig('images/filtered_' + imgs[j].split('.')[0] + '.png', dpi=300)
    plt.show()

