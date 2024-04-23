import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision

import copy

def get_data_loaders_MNIST(train_batch_size, val_batch_size, test_batch_size):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the images
    ])

    # Load the full training dataset
    full_trainset = datasets.MNIST(root='lab0/data', train=True, 
                                   download=True, transform=transform)

    # Split the full training dataset into training and validation
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

    # Load the test dataset
    test_dataset = datasets.MNIST(root='lab0/data', train=False, 
                                  download=True, transform=transform)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader  

def get_data_loaders_SVHN(train_batch_size, val_batch_size, test_batch_size):
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize SVHN images to 28x28
        transforms.Grayscale(num_output_channels=1),  # Convert images to greyscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize the greyscale images
    ])

    # Load the full training dataset
    full_trainset = datasets.SVHN(root='lab0/data', split='train', 
                                  download=True, transform=transform)

    # Split the full training dataset into training and validation
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

    # Load the test dataset
    test_dataset = datasets.SVHN(root='lab0/data', split='test', 
                                 download=True, transform=transform)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=3, stride=1) # 1 in-channels because greyscale (MNIST)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)  # Dropout
        self.dropout2 = nn.Dropout(0.5)   # Dropout
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    



def train_and_validate(model, num_epochs, optimizer, criterion, trainloader, valloader, patience=5):
    print("Training")
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    stop_count = 0  # Initialize the counter for early stopping
    early_stopping_triggered = False


    best_model_state = None  # To store the state of the best model

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 20 == 19:  # Print training loss every 20 mini-batches
                print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / (i+1):.4f}', end="")
        print()
        avg_train_loss = running_loss / len(trainloader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)
        #print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')
        


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())  # Deep copy the model state
            stop_count = 0  # Reset the early stopping counter
            print(f'New best model found at epoch {epoch + 1} with validation loss: {best_val_loss}')
        else:
            stop_count += 1  # Increment the counter if no improvement
            print(f'No improvement in validation loss for epoch {epoch+1}. Early stopping counter: {stop_count}/{patience}')
            
            if stop_count >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.')
                early_stopping_triggered = True
                break  # Break out of the loop to stop training

    # Load the best model state if early stopping was triggered
    if early_stopping_triggered:
        print("Loading the best model state due to early stopping.")
        model.load_state_dict(best_model_state)
        return model
    model.load_state_dict(best_model_state)
    print("Finished Training and Validation")
    return model

def test(model,testloader):
    correct = 0
    total = 0
    model.eval()
    print("Testing")
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'{acc}%')

model = Net()



criterion = nn.NLLLoss(reduction='sum')

#model.load_state_dict(torch.load('model_mnist.pth'))

optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.005) # weight decay - L2 regularization

num_epochs = 50

train_loader_SVHN, val_loader_SVHN, test_loader_SVHN = get_data_loaders_SVHN(64, 64, 64)
train_loader_MNIST, val_loader_MNIST, test_loader_MNIST = get_data_loaders_MNIST(64, 64, 64)

if __name__ == '__main__':
    
    trained_model = train_and_validate(model, num_epochs, optimizer, criterion, train_loader_MNIST, val_loader_MNIST)
    #torch.save(trained_model.state_dict(), 'model_mnist.pth')
    test(trained_model, test_loader_MNIST)
    trained_model = train_and_validate(trained_model, num_epochs, optimizer, criterion, train_loader_SVHN, val_loader_SVHN)
    test(trained_model, test_loader_SVHN)
