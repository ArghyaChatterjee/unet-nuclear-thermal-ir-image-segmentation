import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Define the UNet model architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        enc1 = self.encoder(x)
        dec1 = self.decoder(enc1)
        return dec1

# Define a custom dataset for testing
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = Image.open(image_name).convert('RGB')
        mask = Image.open(mask_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Data transformations (should be the same as used during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Data directories (adjust these paths to your test dataset)
image_dir = 'test_data'
mask_dir = 'gt_test_data'

# Create the custom dataset and data loader for testing
test_dataset = CustomDataset(image_dir, mask_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Batch size set to 1 for testing

# Initialize the UNet model and load the trained weights
model = UNet()
model.load_state_dict(torch.load('unet_weights.pth'))

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define the loss function (same as used during training)
criterion = nn.MSELoss()

# Begin testing
total_loss = 0
with torch.no_grad():  # No need to track gradients for testing
    for images, true_masks in test_dataloader:
        # Move the data to the GPU if available
        images = images.to(device)
        true_masks = true_masks.to(device)

        # Forward pass: compute the output of the model
        predicted_masks = model(images)

        # Calculate the loss
        loss = criterion(predicted_masks, true_masks)
        total_loss += loss.item()

    # Calculate the average loss over all test data
    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss}")

