import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Dinov2WithRegistersForImageClassification,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)
import evaluate
from datasets import Dataset

import torch
from torch.utils.data import random_split
from datasets import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, RandomHorizontalFlip, Resize, CenterCrop
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import TrainingArguments, Trainer, DefaultDataCollator
import evaluate

# 1) Custom dataset definition
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


# Prepare dataset splits
root_dir = Path("/mnt/e/Datasets/Combined-Datasets")
full_dataset = SUNSEGClassificationDataset([d for d in root_dir.iterdir()], sunseg_p_neg=0.3845)
subset_size = int(0.25 * len(full_dataset))
full_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

def to_hf_dict(subset):
    paths = [sample["image"] for sample in subset]
    labels = [sample["label"] for sample in subset]
    return Dataset.from_dict({"image": paths, "label": labels})

hf_train = to_hf_dict(train_subset)
hf_val   = to_hf_dict(val_subset)

# Build label mappings
labels = ["negative", "positive"]
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

# 3) Define transforms (on-the-fly)
checkpoint = "facebook/dinov2-with-registers-small-imagenet1k-1-layer"
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")


train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(Image.open(image).convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(Image.open(image).convert("RGB")) for image in example_batch["image"]]
    return example_batch

hf_train.set_transform(preprocess_train)
hf_val.set_transform(preprocess_val)

# 4) Data collator
data_collator = DefaultDataCollator()

# 5) Metric
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

# 6) Model and Trainer
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

training_args = TrainingArguments(
    output_dir="dinov2_training",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=25,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    processing_class=image_processor,  # For logging & saving image_processor config
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('models/dinov2_combined.pt')
trainer.save_state('models/dinov2_combined_state.pt')
trainer.save_metrics('models/dinov2_combined_metrics.json')
