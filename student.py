#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.

    Method:
        Convolutional Neural Network and Adam optimizers, which is deep neural network training algorithm, 
        uses an adaptive learning rate optimisation mechanism, is used for the task. Because of its deep 
        learning method, takes an image as input, assign priority to distinct items, and distinguish between them.

    Parameters:
        The torch.nn module is used to train the network and the hyperparameters like learning rate=0.004, 
        batch size=128, 100 epochs, and variable train_val_split = 1.

    Transforms:
        We used multiple transforms that help in image processing.
        -transforms.Greyscale which is used to convert the image from RGB to Grayscale means no of channels were 
        reduced to 1 from 3. transforms.
        -RandomHorizontalFlip (flips the image horizontally with a given probability), transforms.
        -RandomAdjustSharpness(Increasing the sharpness helps to highlight the edges clearly) 
        -transforms.ToTensor. (convert a numpy.ndarray to tensor)

    Training:
        6 Convolutional layers are used to extracts different features from the input images at every layer. 
        Every convolution layer is followed by BatchNorm2d, which makes the network faster by normalizing the input 
        layers by rescaling them with the num_features(dimensions) of previous out channel. 
        Relu, the activation function to apply after every layer for the network to be non-linear, and alternate 
        MaxPool2d, is applied that helps to reduce the image size while retaining the important pixel values. 
        Adequate padding helped to calculate the image sizes at every convolution layer also at no data loss while convoluting
        The brute force method helped to decide the number of input and output layers, as we were given the model size of 50 mb
        we reduced the output channels in every convolutional layer as it affects drastically on the model size.

    Loss Function
        The loss Function we used in the model is Cross Entropy loss because it reduces the difference between 
        expected and actual probability distributions. i.e the decision boundaries in the classification task is higher.

    Improving Model
        Introducing Dropout layer and BatchNorm made a significant progress on the accuracy increasing by about 50% 
        for the first epochs but on the other hand we did face an overfitting issue, which was resolved after removing 
        a few transforms and only keeping the most relevant ones.  



"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.ToTensor()
        ])
    elif mode == 'test':
        return transforms.Compose([
        transforms.Grayscale(1),transforms.ToTensor()
    ])

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        #convolution layer 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding = 1)
        self.bn1   = nn.BatchNorm2d(num_features = 32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        #[128, 32, 32, 32]----[Batch size, out channels, Width, Hight]

        #Convolution layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1)
        self.bn2   = nn.BatchNorm2d(num_features = 64)
        self.relu2 = nn.ReLU()
        #[128, 64, 32, 32]

        #Convolution layer 3
        self.conv3 = nn.Conv2d(64, 96, 3, padding = 1)
        self.bn3   = nn.BatchNorm2d(num_features = 96)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        #[128, 96, 16, 16]

        #Convolution layer 4
        self.conv4 = nn.Conv2d(96, 128, 3, padding = 1)
        self.bn4   = nn.BatchNorm2d(num_features = 128)
        self.relu4 = nn.ReLU()
        #[128, 128, 16, 16]

        #Convolution Layer 5
        self.conv5 = nn.Conv2d(128, 196, 3, padding = 1)
        self.bn5   = nn.BatchNorm2d(num_features = 196)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = 2)
        #[128, 196, 8, 8]

        #Convolution layer 6
        self.conv6 = nn.Conv2d(196, 256, 3, padding = 1)
        self.bn6   = nn.BatchNorm2d(num_features = 256)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size = 2)
        #[128, 256, 8, 8]
        self.convIn = 256*4*4
        
        #Input Layers
        self.drop1 = nn.Dropout()
        self.input = nn.Linear((self.convIn), 2048)
        self.bn7   = nn.BatchNorm1d(num_features = 2048)
        self.relu7 = nn.ReLU()
        
        #Hidden Layer
        self.drop2 = nn.Dropout()
        self.hid1  = nn.Linear(2048, 512)
        self.bn8   = nn.BatchNorm1d(num_features = 512)
        self.relu8 = nn.ReLU()

        #Output Layer
        self.outp  = nn.Linear(512, 14)
        self.bn10   = nn.BatchNorm1d(num_features = 14)



    def forward(self, x):

        #Convolution 1
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool1(output)

        #Convolution 2
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)

        #Convolution 3
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.pool3(output)

        #Convolution 4
        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)

        #Convolution 5
        output = self.conv5(output)
        output = self.bn5(output)
        output = self.relu5(output)
        output = self.pool5(output)

        #Convolution 6 
        output = self.conv6(output)
        output = self.bn6(output)
        output = self.relu6(output)
        output = self.pool6(output)
        
        #Flat output for passing in Linear layer
        output = output.view(-1, self.convIn)

        #Input Layer
        output = self.drop1(output)
        output = self.input(output)
        output = self.bn7(output)
        output = self.relu7(output)

        #Hidden Layer
        output = self.drop2(output)
        output = self.hid1(output)
        output = self.bn8(output)
        output = self.relu8(output)

        #Ouput Layer
        output = self.outp(output)
        output = self.bn10(output)
        return output

net = Network()
lossFunc = nn.CrossEntropyLoss()
############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data/"
train_val_split = 1
batch_size = 128
epochs = 100
optimiser = optim.Adam(net.parameters(), lr=0.004)
