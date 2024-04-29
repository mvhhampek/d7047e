import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import sys

import copy

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
channels = 1 # suggested default : 1, number of image channels (gray scale)
img_size = 28 # suggested default : 28, size of each image dimension
img_shape = (channels, img_size, img_size) # (Channels, Image Size(H), Image Size(W))
latent_dim = 100 # suggested default. dimensionality of the latent space
cuda = True if torch.cuda.is_available() else False # GPU Setting

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
from tqdm.notebook import tqdm
import time
# Visualize result
import matplotlib.pyplot as plt



def get_data_loaders_MNIST(train_batch_size, val_batch_size, test_batch_size):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    # Load the full training dataset
    full_trainset = datasets.MNIST(root='lab2/data', train=True, 
                                   download=True, transform=transform)

    # Split the full training dataset into training and validation
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

    # Load the test dataset
    test_dataset = datasets.MNIST(root='lab2/data', train=False, 
                                  download=True, transform=transform)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        def block(input_features, output_features, normalize=True):
            layers = [nn.Linear(input_features, output_features)]
            if normalize: # Default
                layers.append(nn.BatchNorm1d(output_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # inplace=True : modify the input directly. It can slightly decrease the memory usage.
            return layers # return list of layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False), # Asterisk('*') in front of block means unpacking list of layers - leave only values(layers) in list
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, channels*img_size*img_size), # 
            nn.Tanh() # result : from -1 to 1
        )

    def forward(self, z): # z == latent vector(random input vector)
        img = self.model(z) # (64, 100) --(model)--> (64, 784)
        img = img.view(img.size(0), *img_shape) # img.size(0) == N(Batch Size), (N, C, H, W) == default --> (64, 1, 28, 28)
        return img
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size, 512), # (28*28, 512)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # result : from 0 to 1
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1) #flatten -> from (64, 1, 28, 28) to (64, 1*28*28)
        validity = self.model(img_flat) # Discriminate -> Real? or Fake? (64, 784) -> (64, 1)
        return validity
    
def train(num_epochs, trainGenerator, trainDiscriminator, optimizer_G, optimizer_D, adversarial_loss, train_loader):
    best_val_loss = float('inf')
    stop_count = 0
    best_model_state = None
    print("Beginning training")
    for epoch in range(num_epochs): # suggested default = 200

        for i, (data, _) in enumerate(train_loader):

            # Print progress
            sys.stdout.write('\rEpoch [{}/{}] Progress: [{}/{}] {:.2f}%'.format(epoch+1, num_epochs, i + 1, len(train_loader), ((epoch * len(train_loader) + i + 1) / (num_epochs * len(train_loader)) * 100)))
            sys.stdout.flush()
            # Adversarial ground truths 
            valid = Tensor(data.size(0), 1).fill_(1.0)
            fake = Tensor(data.size(0), 1).fill_(0.0) 
            real_data = data.type(Tensor)
            

            # ------------
            # Train Generator
            # ------------
            optimizer_G.zero_grad()
            z = torch.randn(data.shape[0], latent_dim)
            gen_data = trainGenerator(z)
            g_loss = adversarial_loss(trainDiscriminator(gen_data), valid)
            g_loss.backward()
            optimizer_G.step()

            # ------------
            # Train Discriminator
            # ------------
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(trainDiscriminator(real_data), valid) # torch.nn.BCELoss() compare result(64x1) and valid(64x1, filled with 1)
            fake_loss = adversarial_loss(trainDiscriminator(gen_data.detach()), fake) # We are learning the discriminator now. So have to use detach()                                                                
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()# If didn't use detach() for gen_imgs, all weights of the generator will be calculated with backward(). 
            optimizer_D.step()



            # ------------
            # Visualization 
            # ------------

            sample_z_in_train = torch.randn(data.shape[0], latent_dim)
            # z.shape == torch.Size([64, 100])
            sample_gen_imgs_in_train = trainGenerator(sample_z_in_train).detach().cpu()
            # gen_imgs.shape == torch.Size([64, 1, 28, 28])
            
            if ((i+1) % 600) == 0: # show while batch - 200/657, 400/657, 600/657
                
                nrow=1
                ncols=5
                fig, axes = plt.subplots(nrows=nrow,ncols=ncols, figsize=(8,2))
                plt.suptitle('EPOCH : {} | BATCH(ITERATION) : {}'.format(epoch+1, i+1))
                for ncol in range(ncols):
                    axes[ncol].imshow(sample_gen_imgs_in_train.permute(0,2,3,1)[ncol], cmap='gray')
                    axes[ncol].axis('off')
                plt.show()


                
        print(
            " [D loss: %f] [G loss: %f]"
            % (d_loss.item(), g_loss.item())
        )
        
                                             
        
             
        
       

def main():

    # suggested default - beta parameters (decay of first order momentum of gradients)
    b1 = 0.5
    b2 = 0.999

    # suggested default - learning rate
    lr = 0.0002 
    

    generator = Generator()
    discriminator = Discriminator()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1,b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1,b2))
    adversarial_loss = torch.nn.BCELoss()

    train_loader_MNIST, val_loader_MNIST, test_loader_MNIST = get_data_loaders_MNIST(64, 64, 64)

    model=train(5, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, train_loader_MNIST)


if __name__ == "__main__":
    main()