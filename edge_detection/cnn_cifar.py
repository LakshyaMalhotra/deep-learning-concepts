import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler            # used to split data in train and valid sets 
import numpy as np
import matplotlib.pyplot as plt

batch_size = 20

valid_size = 0.2

# defining transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

# choose the train and test datasets
train_data = datasets.CIFAR10('data', train=True, transform=transform, 
                                download=True)
test_data = datasets.CIFAR10('data', train=False, transform=transform, 
                                download=True)

# obtain train indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))

# shuffling the indices randomly
"""
    - Key takeaway: how to use `SubsetRandomSampler`
    - How to create a valid dataset
"""
np.random.shuffle(indices)
split = int(np.floor(num_train * valid_size))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining train and valid batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# create data loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                            sampler=train_sampler)
validloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                            sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

print(len(train_data))
print(len(trainloader))
print(len(validloader))
print(len(testloader))

# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    plt.imshow(np.transpose(img, (1,2,0)))    # convert the image tensor

# obtain one batch of training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# changing images to numpy arrays from torch tensors
images = images.numpy()

fig = plt.figure(figsize=(20,4))
# display 20 images (1 batch)
# for idx in range(images.shape[0]):
#     ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(classes[labels[idx]])

# plt.show()

# define the network arch
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)     # output shape: 16x32x32
        self.conv2 = nn.Conv2d(16, 32, 3)               # output shape: 32x14x14
        self.conv3 = nn.Conv2d(32, 32, 3)               # output shape: 32x5x5

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)        

        self.fc1 = nn.Linear(32 * 5* 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)

        x = x.view(-1, 32*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x

model = Net()
print(model)
print(len(trainloader.dataset))
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the network
epochs = 8

valid_loss_min = np.Inf     # track changes in validation loss
for e in range(1, epochs+1):

    # keep track of train and valid loss
    train_loss = 0.0
    valid_loss = 0.0

    train_acc = 0.0
    valid_acc = 0.0

    model.train()
    for images, labels in trainloader:
        
        # clear the gradients
        optimizer.zero_grad()

        # forward pass: get the output
        output = model(images)

        # calculate the batch loss
        loss = criterion(output, labels)

        # backpropagation: compute gradient w.r.t. model parameters
        loss.backward()

        # perform optimiziation step (updating optimizer)
        optimizer.step()

        # update running loss
        train_loss += loss.item()
        
        # calculate the class with maximum proabability
        max_idx = torch.argmax(output, dim=1)

        # running accuracy
        # top_k, top_class = output.topk(1, dim=1)
        # print(top_class)
        # corrects = (top_class == labels.data.view(*top_class.shape)).float().sum()#.type(torch.FloatTensor)
        corrects = (max_idx == labels.data).float().sum()
        train_acc += (corrects) / images.shape[0]       # average of corrects in a batch

    # validation mode
    model.eval()
    for images, labels in validloader:
         
        # forward pass
        output = model(images)

        # calculate loss
        loss = criterion(output, labels)

        # update validation loss
        valid_loss += loss.item()

        #  calculate the class with max proba
        max_idx = torch.argmax(output, dim=1)
        # running accuarcy
        # top_k, top_class = output.topk(1, dim=1)
        # corrects = (top_class == labels.data.view(*top_class.shape)).float().sum()#.type(torch.FloatTensor)
        corrects = (max_idx == labels.data).float().sum()
        valid_acc += (corrects) / images.shape[0]      # average of corrects in a batch

    train_loss = train_loss / len(trainloader)
    valid_loss = valid_loss / len(validloader)
    train_acc = 100 * train_acc / (len(trainloader))
    valid_acc = 100 * valid_acc / len(validloader)

    print("Epoch: {} \t Training loss: {:.6f} \t Validation loss: {:.6f}".format(
        e, train_loss, valid_loss
    ))
    print("Train accuracy: {:.3f} \t Validation accuracy: {:.3f}".format(
        train_acc, valid_acc))

    if valid_loss <= valid_loss_min:
        print("Validation loss decreased ({:.6f} --> {:.6f}). Saving model ... ".format(
            valid_loss_min, valid_loss))
        torch.save(model.state_dict(), 'cifar_model.pt')
        valid_loss_min = valid_loss