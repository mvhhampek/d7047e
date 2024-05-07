import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import tqdm
from tqdm.notebook import trange, tqdm
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)


# training image shape, 60 000 x 784

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding='same')
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # Conv Layer 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        # Conv Layer 2
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # Flatten the data for the dense layers
        x = x.view(-1, 7*7*64)
        # Dense Layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Output Layer
        x = self.fc2(x)
        return x

# Initialize the model and move it to the device
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# Training loop
model.train()
# Train the model

# Variables to calculate average loss and accuracy
total_loss = 0
correct = 0
total = 0
total_batches = 0  # Total batches processed

# Loop to train the model for a total of 1000 batches
while total_batches < 1000:
    for batch_idx, (data, target) in enumerate(train_loader):
        if total_batches >= 1000:  # Stop after 1000 batches
            break
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = output.argmax(dim=1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
        total_batches += 1  # Increment total_batches

        if (total_batches % 200 == 0 or total_batches == 1000):  # Report every 200 batches and at the 1000th batch
            train_accuracy = 100 * correct / total
            average_loss = total_loss / 200
            print(f"Step {total_batches}, Avg Loss: {average_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
            total_loss = 0  # Reset the loss for the next 200 batches
            correct = 0
            total = 0

test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        correct = output.argmax(dim=1).eq(target).sum().item()
        test_accuracy = correct / data.shape[0]
        print(f"test accuracy {test_accuracy:.3f}")

        break  # Only evaluate the first batch



def get_prediction(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        output = model(image)
        _, predicted = torch.max(output, 1)
        probabilities = torch.softmax(output, dim=1)
        return predicted.item(), probabilities.squeeze().cpu().numpy()  # Return predicted label and output probabilities

def plot_images_with_predictions(model, test_loader, num_images=9):
    model.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()

        for i, (data, target) in enumerate(test_loader):
            if i >= num_images:
                break
            image = data[0]
            label = target[0].item()
            prediction, probabilities = get_prediction(model, image)
            
            # Plot the image
            ax = axes[i]
            ax.imshow(image.permute(1, 2, 0).cpu())
            ax.set_title(f"Predicted: {prediction}, Actual: {label}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()



plot_images_with_predictions(model, test_loader)


def plot_predictions(image_list, output_probs=False, adversarial=False):
    '''
    Evaluate images against trained model and plot images.
    If adversarial == True, replace middle image title appropriately
    Return probability list if output_probs == True
    '''
    prob = model(image_list)
    prob = torch.softmax(prob, dim=1)  # Apply softmax normalization
    
    pred_list = torch.argmax(prob, dim=1)
    pct_list = torch.max(prob, dim=1)[0] * 100  # Ensure to multiply by 100 for percentage

    # Setup image grid
    import math
    cols = 3
    rows = math.ceil(image_list.shape[0]/cols)
    fig = plt.figure(1, (12., 12.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates grid of axes
                     axes_pad=0.5,  # pad between axes in inch.
                     )
    
    # Get probs, images and populate grid
    for i in range(len(prob)):
        pred_label = pred_list[i].item()
        certainty = pct_list[i].item()  # No need to multiply by 100 here

        image = image_list[i].detach().cpu().numpy().reshape(28,28)
        grid[i].imshow(image)
        
        grid[i].set_title('Label: {0} \nCertainty: {1:.2f}%' \
                          .format(pred_label, 
                                  certainty))
        
        # Only use when plotting original, partial deriv and adversarial images
        if (adversarial) & (i % 3 == 1): 
            grid[i].set_title("Adversarial \nPartial Derivatives")
        
    plt.show()
    
    return prob if output_probs else None




def create_plot_adversarial_images(x_image, y_label, model, lr=0.1, n_steps=1, output_probs=False):
    
    original_image = x_image.clone().detach()
    probs_per_step = []
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([x_image.requires_grad_()], lr=lr)
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = model(x_image)
        loss = criterion(outputs, y_label)
        loss.backward()
        optimizer.step()
        
        # Clip the image to prevent invalid values
        x_image.data = torch.clamp(x_image.data, 0, 1)
        
        # Print/plot images and return probabilities
        img_adv_list = torch.cat((original_image, x_image.grad.data, x_image), dim=0)
        probs = plot_predictions(img_adv_list, output_probs=output_probs, adversarial=True)
        probs_per_step.append(probs) if output_probs else None
    
    return probs_per_step

# Pick a random 2 image from first 1000 images 
index_of_4s = np.where(np.array(test_dataset.targets) == 4)[0][:1000]
rand_index = np.random.randint(0, len(index_of_4s))
image_norm = test_dataset[index_of_4s[rand_index]][0].unsqueeze(0).to(device)

# Create adversarial label (target label 9)
label_adv = torch.zeros(1, 10).to(device)
label_adv[0, 9] = 1  # Change index to 9 for the target label


create_plot_adversarial_images(image_norm, label_adv, model, lr=1, n_steps=10)
