import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set device to CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Define the model selector function
def get_model(model_type, num_classes):
    if model_type == "shallow":
        # Shallow CNN
        class ShallowCNN(nn.Module):
            def __init__(self):
                super(ShallowCNN, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.fc = nn.Linear(16 * 64 * 64, num_classes)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return ShallowCNN().to(device)

    elif model_type == "deep":
        # Deep CNN
        class DeepCNN(nn.Module):
            def __init__(self):
                super(DeepCNN, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.fc = nn.Sequential(
                    nn.Linear(64 * 32 * 32, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        return DeepCNN().to(device)

    elif model_type == "resnet":
        # Pretrained ResNet
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, num_classes)
        return model.to(device)

    else:
        raise ValueError("Invalid model type! Choose 'shallow', 'deep', or 'resnet'.")

# Custom collate function to handle variable-sized batches
def custom_collate_fn(batch):
    images, targets = [], []
    for sample in batch:
        # Skip invalid samples or empty annotations
        if sample[1]:
            images.append(sample[0])  # Append the image
            targets.append(sample[1])  # Append the annotations
    if not images:  # If all samples are invalid, return empty tensors
        return torch.empty(0), []
    return torch.stack(images), targets

if __name__ == "__main__":
    # Define dataset paths
    dataset_path = "./self_driving_car_dataset.coco/export/_annotations.coco.json"  # Path to the annotations file
    images_path = "./self_driving_car_dataset.coco/export"  # Path to the images folder

    # Define transformations for preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a uniform size
        transforms.ToTensor(),  # Convert images to PyTorch tensors
    ])

    # Load the COCO dataset
    dataset = CocoDetection(
        root=images_path,  # Path to the folder containing images
        annFile=dataset_path,  # Path to the annotations file
        transform=transform  # Apply preprocessing transformations
    )

    # Split dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each dataset split
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    # Get number of classes from the dataset
    num_classes = len(dataset.coco.getCatIds())

    # Choose the model type: "shallow", "deep", or "resnet"
    model_type = "resnet"  # Change this to "deep" or "resnet" as needed
    model = get_model(model_type, num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):  # Replace 10 with the number of epochs
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, targets in train_loader:
            if len(images) == 0:  # Skip empty batches
                continue

            # Extract labels from COCO annotations
            labels = []
            for target in targets:
                if isinstance(target, list) and target:  # Check if target is a non-empty list
                    labels.append(target[0]['category_id'])  # Use the first category_id in the list
                else:
                    labels.append(0)  # Default to class 0 if no annotations are present
            labels = torch.tensor(labels).to(device)

            # Move images to the specified device
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation loop
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, targets in test_loader:
            if len(images) == 0:  # Skip empty batches
                continue

            # Extract labels from COCO annotations
            labels = []
            for target in targets:
                if isinstance(target, list) and target:  # Check if target is a non-empty list
                    labels.append(target[0]['category_id'])  # Use the first category_id in the list
                else:
                    labels.append(0)  # Default to class 0 if no annotations are present
            labels = torch.tensor(labels).to(device)

            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
