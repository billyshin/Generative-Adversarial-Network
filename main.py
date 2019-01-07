"""
Generative Adversarial Network (GAN)
Deep Convolutional GANs
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Hyperparameters
batchSize = 64
imageSize = 64

# Creating the transformations
# We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) 


# ==================================================================== Loading Data ===============================================================================
# Loading the dataset (we use CIFAR10 as our dataset in this case)
dataset = dset.CIFAR10(root = './data', download = True, transform = transform) 
# We use dataLoader to get the images of the training set batch by batch.
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 


# ============================================================ Defining Weight Initialization Function ============================================================
"""
A function that takes as input a neural network m and that will initialize all its weights.
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ==================================================================== Defining the Generator =====================================================================
"""
A clss that defines the generator.
"""
class G(nn.Module): 
    def __init__(self): 
        # Inherit from the nn.Module tools
        super(G, self).__init__() 
        # Create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, ...)
        self.main = nn.Sequential( 
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False), # Start with an inversed convolution
            nn.BatchNorm2d(512), # Normalize all the features along the dimension of the batch
            nn.ReLU(True), # Apply a ReLU rectification to break the linearity
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False), # Add another inversed convolution
            nn.BatchNorm2d(256), # Normalize again
            nn.ReLU(True), # Apply another ReLU
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False), # Add another inversed convolution
            nn.BatchNorm2d(128), # Normalize again
            nn.ReLU(True), # Apply another ReLU
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False), # Add another inversed convolution
            nn.BatchNorm2d(64), # Normalize again
            nn.ReLU(True), # Apply another ReLU
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False), # Add another inversed convolution
            nn.Tanh() # Apply a Tanh rectification to break the linearity and stay between -1 and +1
        )

    """
    The forward function that takes as argument an input that will be fed to the neural network, and that will return the output containing the generated images.
    """
    def forward(self, input):
        # Forward propagate the signal through the whole neural network of the generator defined by self.main
        output = self.main(input) 
        # Return the output containing the generated images
        return output 


# ==================================================================== Creating the Generator =====================================================================
netG = G()
# Initialize all the weights of its neural network
netG.apply(weights_init) 


# ================================================================== Defining the Discriminator ===================================================================
"""
A class to define the discriminator.
"""
class D(nn.Module): 
    def __init__(self): 
        # Inherit from the nn.Module tools
        super(D, self).__init__() 
        self.main = nn.Sequential( # Create a meta module of a neural network that will contain a sequence of modules (convolutions, full connections, etc.)
            nn.Conv2d(3, 64, 4, 2, 1, bias = False), # Start with a convolution
            nn.LeakyReLU(0.2, inplace = True), # Apply a LeakyReLU
            nn.Conv2d(64, 128, 4, 2, 1, bias = False), # Add another convolution
            nn.BatchNorm2d(128), # Normalize all the features along the dimension of the batch
            nn.LeakyReLU(0.2, inplace = True), # Apply another LeakyReLU
            nn.Conv2d(128, 256, 4, 2, 1, bias = False), # Add another convolution
            nn.BatchNorm2d(256), # Normalize again
            nn.LeakyReLU(0.2, inplace = True), # Apply another LeakyReLU
            nn.Conv2d(256, 512, 4, 2, 1, bias = False), # Add another convolution
            nn.BatchNorm2d(512), # Normalize again
            nn.LeakyReLU(0.2, inplace = True), # Apply another LeakyReLU
            nn.Conv2d(512, 1, 4, 1, 0, bias = False), # Add another convolution
            nn.Sigmoid() # Apply a Sigmoid rectification to break the linearity and stay between 0 and 1
        )

    """
    The forward function that takes as argument an input that will be fed to the neural network, and that will return the output which will be a value between 0 and 1.
    """
    def forward(self, input): 
        # Forward propagate the signal through the whole neural network of the discriminator defined by self.main
        output = self.main(input) 
        # Return the output which will be a value between 0 and 1
        return output.view(-1) 


# ================================================================== Creating the Discriminator ===================================================================
netD = D() 
# Initialize all the weights of its neural network
netD.apply(weights_init) 