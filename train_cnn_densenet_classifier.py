import random
import torch
import timm
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from monai.data import Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.v2 as T
from PIL import Image

import random
from pathlib import Path
import torch

import random
from pathlib import Path
import torch


class SUNSEGClassificationDataset(torch.utils.data.Dataset):
    """
    A dataset that collects (image_path, label) pairs.
    For the sunseg dataset, images are grouped into non-overlapping pairs.
    The sampling percentage is applied uniformly to these pairs, and from each selected pair,
    both images are added individually as samples.
    Non-sunseg datasets return individual images as samples.
    """

    def __init__(self, datasets: list[Path], sunseg_p_pos=1.0, sunseg_p_neg=1.0, subsample=1):
        self.samples = []
        for dataset in datasets:
            pos_dir = dataset / 'positive'
            neg_dir = dataset / 'negative'
            if dataset.stem == "sunseg":
                # Process sunseg positive images in pairs.
                if pos_dir.is_dir():
                    for case in pos_dir.iterdir():
                        images = sorted(list(case.iterdir()))
                        if len(images) < 2:
                            continue
                        # Group into non-overlapping pairs: (img0, img1), (img2, img3), ...
                        pairs = list(zip(images[0::2], images[1::2]))
                        num_pairs = max(1, int(len(pairs) * sunseg_p_pos))
                        if num_pairs == 1:
                            selected_indices = [0]
                        else:
                            selected_indices = [round(i * (len(pairs) - 1) / (num_pairs - 1)) for i in range(num_pairs)]
                        for idx in selected_indices:
                            pair = pairs[idx]
                            # Add each image in the pair as a separate sample.
                            self.samples.append((pair[0], 1))
                            self.samples.append((pair[1], 1))
                # Process sunseg negative images in pairs.
                if neg_dir.is_dir():
                    for case in neg_dir.iterdir():
                        images = sorted(list(case.iterdir()))
                        if len(images) < 2:
                            continue
                        pairs = list(zip(images[0::2], images[1::2]))
                        num_pairs = max(1, int(len(pairs) * sunseg_p_neg))
                        if num_pairs == 1:
                            selected_indices = [0]
                        else:
                            selected_indices = [round(i * (len(pairs) - 1) / (num_pairs - 1)) for i in range(num_pairs)]
                        for idx in selected_indices:
                            pair = pairs[idx]
                            self.samples.append((pair[0], 0))
                            self.samples.append((pair[1], 0))
            else:
                # For non-sunseg datasets, use all images individually.
                if pos_dir.is_dir():
                    for case in pos_dir.iterdir():
                        for image in case.iterdir():
                            self.samples.append((image, 1))
                if neg_dir.is_dir():
                    for case in neg_dir.iterdir():
                        for image in case.iterdir():
                            self.samples.append((image, 0))

        if subsample > 1:
            self.samples = self.samples[::subsample]

        self.classes = ['negative', 'positive']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        return {"image": str(image_path), "label": label}


# ---------------------------
# 1) Create the full dataset
# ---------------------------
root_dir = Path("/mnt/e/Datasets/Combined-Datasets")
full_dataset = SUNSEGClassificationDataset([d for d in root_dir.iterdir()], sunseg_p_neg=0.3845)

# ---------------------------
# 2) Split into train/val
# ---------------------------
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

print(f"Train size: {len(train_subset)}")
print(f"Val size: {len(val_subset)}")
print(f"Total size: {len(full_dataset)}")
print(f"Positive samples: {sum(1 for x in full_dataset if x['label'] == 1)}")
print(f"Negative samples: {sum(1 for x in full_dataset if x['label'] == 0)}")

# ---------------------------
# 3) Prepare transforms
# ---------------------------
model = timm.create_model('densenet121.ra_in1k', pretrained=True)
data_config = timm.data.resolve_model_data_config(model)

train_transforms = timm.data.create_transform(
    **data_config,
    is_training=True
)
val_transforms = timm.data.create_transform(
    **data_config,
    is_training=False
)

train_dataset = Dataset(data=train_subset, transform=lambda x: {
    "image": train_transforms(Image.open(x["image"]).convert("RGB")),
    "label": x["label"]
})
val_dataset = Dataset(data=val_subset, transform=lambda x: {
    "image": val_transforms(Image.open(x["image"]).convert("RGB")),
    "label": x["label"]
})

# ---------------------------
# 4) Create DataLoaders
# ---------------------------
batch_size = 32
num_workers = 4

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ---------------------------
# 5) Modify the model for binary classification
# ---------------------------
# Densenet-121 in timm often uses 'classifier' as the final layer
model.classifier = nn.Linear(model.classifier.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 50

# ---------------------------
# 6) Training loop with tqdm
# ---------------------------
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss, val_correct = 0.0, 0

    for batch in tqdm(val_loader, desc="Validating"):
        with torch.no_grad():
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss   = val_loss / len(val_loader)
    val_accuracy   = val_correct / len(val_subset)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f} "
          f"Val Acc: {val_accuracy:.4f}")

    # Save model if validation accuracy improves
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"models/best_model_densenet121_epoch_{epoch+1}_balanced_dataset.pt")

print("Training complete. Best validation accuracy:", best_val_acc)
