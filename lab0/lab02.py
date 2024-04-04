from torchvision.models import alexnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

def get_data_loaders(train_batch_size, val_batch_size, test_batch_size):
    # Define the transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the full training dataset
    full_trainset = torchvision.datasets.CIFAR10(root='lab0/data', train=True,
                                                 download=True, transform=transform)

    # Split the full training dataset into training and validation
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])

    # Load the test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='lab0/data', train=False,
                                                download=True, transform=transform)

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



def train_and_validate(model, num_epochs, optimizer, criterion, trainloader, valloader, log_dir=r"lab0/logs"):
    print("Training")
    best_val_loss = float('inf')  # Initialize best validation loss as infinity

    writer = SummaryWriter(log_dir=log_dir)

    best_model_state = None
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
            
            # Print the training loss every 20 mini-batches; this updates the same line
            if i % 20 == 19:  # Adjust the modulus value to change frequency
                print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / (i+1):.4f}', end="")
        print()
        avg_train_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')
        writer.add_scalar('Training Loss', avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        
        # Save the model if it has the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()  # Deep copy the model state
            print(f'New best model found at epoch {epoch + 1} with validation loss: {best_val_loss}')
    
    model.load_state_dict(best_model_state)
    print("Finished Training and Validation")
    writer.close()  # Close the summary writer
    return model



model_alex = alexnet(pretrained = True)

# Freezes the earlier layers of the model, feature extraction

# for param in model_alex.parameters():
#     param.requires_grad = False


# # Add an extra fully connected layer to the classifier
num_ftrs = model_alex.classifier[6].in_features
extra_layer = nn.Linear(num_ftrs, 100)  # Add an extra layer with 100 neurons
model_alex.classifier[6] = nn.Linear(1000, 10)  # Replace the last layer with 10 output neurons for CIFAR-10



criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_alex.parameters(), lr = 0.0001)


train_loader, val_loader, test_loader = get_data_loaders(64, 64, 64)

if __name__ == '__main__':
    trained_model = train_and_validate(model_alex, 50, optimizer, criterion, train_loader, val_loader)
    test(trained_model, test_loader)
