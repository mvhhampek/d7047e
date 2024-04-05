from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time

def get_data_loaders(train_batch_size, val_batch_size, test_batch_size, transform = False):
    # Define the transformation
    if not transform:
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


def train_and_validate(model, num_epochs, optimizer, criterion, trainloader, valloader, log_dir=r"lab0/logs", patience=5):
    print("Training")
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    stop_count = 0
    early_stop = False

    writer = SummaryWriter(log_dir=log_dir)
    best_model_state = None

    for epoch in range(num_epochs):
        start_time = time.time()  # Initialize start time
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
            elapsed_time = time.time() - start_time
            
            if i % 20 == 19:  # Print every 20 mini-batches
                print(f'\rEpoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / (i+1):.4f} \t Elapsed time = {elapsed_time:.2f} seconds', end="")
        print()
        avg_train_loss = running_loss / len(trainloader)
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
        writer.add_scalar('Validation Loss', avg_val_loss, epoch)
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()  # Deep copy the model state
            stop_count = 0  # Reset count
            print(f'New best model found at epoch {epoch + 1} with validation loss: {best_val_loss}')
        else:
            stop_count += 1  # Increment count if no improvement

        if stop_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. No improvement in validation loss for {patience} consecutive epochs.")
            early_stop = True
            break  # Stop training

    if not early_stop:
        model.load_state_dict(best_model_state)  # Load best model state if early stopping wasn't triggered

    print("Finished Training and Validation")
    writer.close()
    return model
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
alexnet_fe = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)




alexnet.classifier = nn.Sequential(
    *list(alexnet.classifier.children())[:-1],  # Retain all but the last original classifier layer
    nn.Linear(4096, 10)  # Add your new layer correctly expecting 4096 inputs, outputting 10 classes
)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(alexnet.parameters(), lr = 0.0001)


for param in alexnet_fe.parameters():
    param.requires_grad = False

alexnet_fe.classifier = nn.Sequential(
    *list(alexnet_fe.classifier.children()),  # Original layers, now frozen
    nn.Linear(4096, 10)  # New layer, by default has requires_grad=True
)

optimizer_fe = optim.Adam(alexnet_fe.classifier[-1].parameters(), lr=0.0001)

batch_size = 64

train_loader, val_loader, test_loader = get_data_loaders(batch_size, batch_size, batch_size, transform = tf)

if __name__ == '__main__':
    start_time = time.time()
    trained_model = train_and_validate(alexnet, 50, optimizer, criterion, train_loader, val_loader)
    test(trained_model, test_loader)
