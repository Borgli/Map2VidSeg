import random
from pathlib import Path
from collections import Counter
import torch

class SUNSEGClassificationDataset(torch.utils.data.Dataset):
    """
    A dataset that collects all (image_path, label) pairs.
    For the sunseg dataset, separate sampling percentages for positive and negative images are allowed.
    """
    def __init__(self, datasets: list[Path], sunseg_p_pos=1.0, sunseg_p_neg=1.0, subsample=1):
        self.samples = []
        for dataset in datasets:
            pos_dir = dataset / 'positive'
            neg_dir = dataset / 'negative'
            if dataset.stem == "sunseg":
                # Apply separate sampling percentages for sunseg
                if pos_dir.is_dir():
                    pos_images = []
                    for case in pos_dir.iterdir():
                        pos_images.extend(list(case.iterdir()))
                    pos_sample_count = max(1, int(len(pos_images) * sunseg_p_pos))
                    pos_sample = random.sample(pos_images, pos_sample_count)
                    for image in pos_sample:
                        self.samples.append((image, 1))
                if neg_dir.is_dir():
                    neg_images = []
                    for case in neg_dir.iterdir():
                        neg_images.extend(list(case.iterdir()))
                    neg_sample_count = max(1, int(len(neg_images) * sunseg_p_neg))
                    neg_sample = random.sample(neg_images, neg_sample_count)
                    for image in neg_sample:
                        self.samples.append((image, 0))
            else:
                # Use all images from non-sunseg datasets
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

def get_distribution_for_dataset(dataset: Path, sampling_percentage=1.0):
    pos_dir = dataset / 'positive'
    neg_dir = dataset / 'negative'
    counts = Counter()
    if dataset.stem == "sunseg":
        if pos_dir.is_dir():
            pos_images = []
            for case in pos_dir.iterdir():
                pos_images.extend(list(case.iterdir()))
            pos_count = max(1, int(len(pos_images) * sampling_percentage))
        else:
            pos_count = 0
        if neg_dir.is_dir():
            neg_images = []
            for case in neg_dir.iterdir():
                neg_images.extend(list(case.iterdir()))
            neg_count = max(1, int(len(neg_images) * sampling_percentage))
        else:
            neg_count = 0
        counts[1] = pos_count
        counts[0] = neg_count
    else:
        if pos_dir.is_dir():
            pos_count = sum(1 for case in pos_dir.iterdir() for _ in case.iterdir())
        else:
            pos_count = 0
        if neg_dir.is_dir():
            neg_count = sum(1 for case in neg_dir.iterdir() for _ in case.iterdir())
        else:
            neg_count = 0
        counts[1] = pos_count
        counts[0] = neg_count
    return counts

def compute_recommended_sunseg_percentages(non_sunseg_counts, sunseg_dataset: Path):
    """
    Computes recommended sampling percentages for sunseg.
    Here, we keep all sunseg positive samples (p_pos = 1.0) and adjust negatives (p_neg) so that:
      non_pos + total_pos == non_neg + p_neg * total_neg
    """
    pos_dir = sunseg_dataset / 'positive'
    neg_dir = sunseg_dataset / 'negative'
    total_pos = sum(len(list(case.iterdir())) for case in pos_dir.iterdir()) if pos_dir.is_dir() else 0
    total_neg = sum(len(list(case.iterdir())) for case in neg_dir.iterdir()) if neg_dir.is_dir() else 0

    non_pos = non_sunseg_counts[1]
    non_neg = non_sunseg_counts[0]

    p_pos = 1.0  # Keep all positives
    desired_total = non_pos + total_pos  # Target overall positive count
    # Solve: non_neg + p_neg * total_neg = desired_total
    if total_neg > 0:
        p_neg = (desired_total - non_neg) / total_neg
        p_neg = max(0.0, min(1.0, p_neg))
    else:
        p_neg = 1.0

    return p_pos, p_neg

if __name__ == "__main__":
    root_dir = Path("/mnt/e/Datasets/Combined-Datasets")
    dataset_paths = [d for d in root_dir.iterdir() if d.is_dir()]

    overall_counts = Counter()
    per_dataset_counts = {}
    non_sunseg_counts = Counter()
    sunseg_dataset = None

    for dataset in dataset_paths:
        if dataset.stem == "sunseg":
            counts = get_distribution_for_dataset(dataset, sampling_percentage=0.3845)
            sunseg_dataset = dataset
        else:
            counts = get_distribution_for_dataset(dataset)
            non_sunseg_counts.update(counts)
        per_dataset_counts[dataset.stem] = counts
        overall_counts.update(counts)

    print("Overall Distribution (negative: 0, positive: 1):", overall_counts)
    print("\nDistribution per dataset:")
    for name, counts in per_dataset_counts.items():
        print(f"  {name}: {dict(counts)}")

    if sunseg_dataset is not None:
        p_pos, p_neg = compute_recommended_sunseg_percentages(non_sunseg_counts, sunseg_dataset)
        print("\nRecommended sunseg sampling percentages:")
        print("  Positive: {:.2f}%".format(p_pos * 100))
        print("  Negative: {:.2f}%".format(p_neg * 100))
    else:
        print("\nNo sunseg dataset found.")

    # Create the full dataset using the computed sampling percentages for sunseg
    full_dataset = SUNSEGClassificationDataset(dataset_paths, sunseg_p_pos=p_pos, sunseg_p_neg=p_neg)
    new_counts = Counter(sample["label"] for sample in full_dataset)
    print("\nNew Overall Distribution after sampling (negative: 0, positive: 1):", new_counts)
