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


# ====================================================================== Training the DCGANs ======================================================================
# Create a criterion object that will measure the error between the prediction and the target
criterion = nn.BCELoss() 
# Create the optimizer object of the discriminator
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 
# Create the optimizer object of the generator
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999)) 

# Iterate over 25 epochs
for epoch in range(25):
     # Iterate over the images of the dataset
    for i, data in enumerate(dataloader, 0):
        print(i)
        # 1st Step: Updating the weights of the neural network of the discriminator
        
        # Initialize to 0 the gradients of the discriminator with respect to the weights
        netD.zero_grad() 
        
        # Training the discriminator with a real image of the dataset
         # Get a real image of the dataset which will be used to train the discriminator
        real, _ = data
        # Wrap it in a variable
        input = Variable(real) 
         # Get the target
        target = Variable(torch.ones(input.size()[0]))
        # Forward propagate this real image into the neural network of the discriminator to get the prediction (a value between 0 and 1)
        output = netD(input) 
         # Compute the loss between the predictions (output) and the target (equal to 1)
        errD_real = criterion(output, target)
        
        # Training the discriminator with a fake image generated by the generator
        # Make a random input vector (noise) of the generator
        noise = Variable(torch.randn(input.size()[0], 100, 1, 1)) 
        # Forward propagate this random input vector into the neural network of the generator to get some fake generated images
        fake = netG(noise) 
        # Get the target
        target = Variable(torch.zeros(input.size()[0])) 
        # Forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1)
        output = netD(fake.detach()) 
        # Compute the loss between the prediction (output) and the target (equal to 0)
        errD_fake = criterion(output, target) 

        # Backpropagating the total error
        # Compute the total error of the discriminator
        errD = errD_real + errD_fake 
        # Backpropagate the loss error by computing the gradients of the total error with respect to the weights of the discriminator
        errD.backward() 
        # Apply the optimizer to update the weights according to how much they are responsible for the loss error of the discriminator
        optimizerD.step() 

        # 2nd Step: Updating the weights of the neural network of the generator

        # Initialize to 0 the gradients of the generator with respect to the weights
        netG.zero_grad() 
         # Get the target
        target = Variable(torch.ones(input.size()[0]))
        # Forward propagate the fake generated images into the neural network of the discriminator to get the prediction (a value between 0 and 1)
        output = netD(fake) 
        # Compute the loss between the prediction (output between 0 and 1) and the target (equal to 1)
        errG = criterion(output, target) 
        # Backpropagate the loss error by computing the gradients of the total error with respect to the weights of the generator
        errG.backward() 
        # Apply the optimizer to update the weights according to how much they are responsible for the loss error of the generator
        optimizerG.step() 
        
        # 3rd Step: Printing the losses and saving the real images and the generated images of the minibatch every 100 steps

        # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])) # We print les losses of the discriminator (Loss_D) and the generator (Loss_G).
        # Every 100 steps
        if i % 100 == 0: 
            # Save the real images of the minibatch
            vutils.save_image(real, '%s/real.png' % "./results", normalize = True) 
            # Get fake generated images
            fake = netG(noise) 
            # Save the fake generated images of the minibatch
            vutils.save_image(fake.data, '%s/fake_epoch_%03d.png' % ("./results", epoch), normalize = True) 
