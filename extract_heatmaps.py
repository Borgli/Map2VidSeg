from pathlib import Path

import numpy
from tqdm import tqdm
from pytorch_grad_cam import GradCAMPlusPlus, HiResCAM




def use_densenet(frames, outdir):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import cv2
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import random_split
    from tqdm import tqdm
    from monai.data import Dataset, DataLoader

    # pytorch-grad-cam
    from pytorch_grad_cam import GradCAMElementWise
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    import supervision as sv

    # 1) Load your timm-based EfficientNet model
    import timm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir.mkdir(parents=True, exist_ok=True)

    model = timm.create_model('densenet121.ra_in1k', pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model_path = "pretraining/models/best_model_densenet121_epoch_6_balanced_dataset.pt"  # e.g., your saved model file
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    categories = ["negative", "positive"]  # or your actual class names

    data_config = timm.data.resolve_model_data_config(model)

    val_transforms = timm.data.create_transform(
        **data_config,
        is_training=False
    )

    val_dataset = Dataset(data=frames, transform=lambda x: {
        "image": val_transforms(Image.open(x["image"]).convert("RGB")),
        "label": x['label'],
        "image_path": x["image_path"]
    })

    # 2) Prepare any transforms you used (resize, normalize, etc.)
    # import torchvision.transforms as T
    #
    # transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],  # example mean/std
    #                 std=[0.229, 0.224, 0.225])
    # ])

    # 3) Example usage: create CAM for a single image
    #    or inside your loop for multiple images

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    cam_outdir = outdir / "cam"
    cam_outdir.mkdir(exist_ok=True)

    heatmap_outdir = outdir / "heatmap"
    heatmap_outdir.mkdir(exist_ok=True)

    confidence_scores = []
    for batch in tqdm(val_loader, desc="Generating CAM DenseNet121"):


        with torch.no_grad():
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            image_paths = batch["image_path"]
            if Path(f"{heatmap_outdir}/{Path(image_paths[0]).name}").exists() and Path(
                    f"{cam_outdir}/{Path(image_paths[0]).stem}.png").exists():
                continue
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(probs)
            print("Predicted class:", categories[pred_class], "with confidence:", probs[pred_class])
            confidence_scores.append({image_paths[0]: probs[pred_class]})

        # test_image_path = Path("some_image.jpg")
        img_pil = Image.open(image_paths[0]).convert("RGB")
        img_pil = img_pil.resize((224, 224))
        img_np = np.array(img_pil, dtype=np.float32) / 255.0  # for visualization later
        # img_tensor = transform(img_pil).unsqueeze(0).to(device)

        # 4) Forward pass to get predictions

        # 5) Grad-CAM: pick a target layer in EfficientNet
        #    For EfficientNet from timm, blocks[-1] is a typical choice
        target_layer = [model.features[-1]]

        # 6) Generate the CAM using GradCAMElementWise
        with HiResCAM(model=model, target_layers=target_layer) as cam:
            grayscale_cam = cam(
                input_tensor=images[0].unsqueeze(0).to(device),
                targets=[ClassifierOutputTarget([1])],
                eigen_smooth=False,
                aug_smooth=False
            )
            # grayscale_cam: [batch_size, H, W] => we take the first index
            grayscale_cam = grayscale_cam[0]

        # 7) Overlay the CAM on the original image
        result_cam = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        cv2.imwrite(f"{heatmap_outdir}/{Path(image_paths[0]).name}", cv2.cvtColor(result_cam, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{cam_outdir}/{Path(image_paths[0]).stem}.png", (grayscale_cam * 255).astype(np.uint8))


def use_dino(frames, outdir):
    import argparse
    import os
    import random

    import cv2
    import torch
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from pytorch_grad_cam.utils.image import show_cam_on_image  # for overlaying heatmaps
    import torch.nn as nn  # for interpolation

    parser = argparse.ArgumentParser(description="Visualize Combined Attention Maps on TrainDataset images")
    parser.add_argument("--dataset", type=str, default="/mnt/e/Datasets/SUN-SEG/TrainDataset",
                        help="Path to TrainDataset folder")
    parser.add_argument("--checkpoint", type=str, default="pretraining/models/dinov2_combined.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output", type=str, default="attn_outputs",
                        help="Directory to save attention outputs")
    parser.add_argument("--max_images", type=int, default=30,
                        help="Maximum number of images to process (default: 10)")
    args = parser.parse_args()

    entries = frames

    # Load image processor for normalization details from the checkpoint's config
    # checkpoint_name = "facebook/dinov2-with-registers-small-imagenet1k-1-layer"
    image_processor = AutoImageProcessor.from_pretrained(args.checkpoint, use_fast=True)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    # if "height" in image_processor.size:
    #    size = (image_processor.size["height"], image_processor.size["width"])
    #    crop_size = size
    # elif "shortest_edge" in image_processor.size:
    #    size = image_processor.size["shortest_edge"]
    #    crop_size = (size, size)

    size = (224, 224)
    crop_size = size

    transforms_val = Compose([
        Resize(size),
        CenterCrop(crop_size),
        ToTensor(),
        normalize,
    ])

    # Load the trained model and specify the eager attention implementation to avoid warnings.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageClassification.from_pretrained(
        args.checkpoint, ignore_mismatched_sizes=True, attn_implementation="eager"
    )
    model.to(device)
    model.eval()

    cam_dir = Path(outdir / 'cam')
    cam_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir = Path(outdir / 'heatmap')
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    for frame_name, label, img_path in tqdm(((e['image'], e['label'], e['image_path']) for e in entries), total=len(entries), desc="Generating CAM DenseNet121"):
        if Path(f"{heatmap_dir}/{Path(img_path).name}").exists() and Path(f"{cam_dir}/{Path(img_path).stem}.png").exists():
            continue

        pil_img = Image.open(img_path).convert("RGB")
        original_img = np.array(pil_img, dtype=np.float32) / 255.0

        # Preprocess image for the model
        img_tensor = transforms_val(pil_img).unsqueeze(0).to(device)

        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = model(img_tensor, output_attentions=True)
        # Get attention from the last layer; shape: (1, num_heads, seq_len, seq_len)
        attentions_all = outputs.attentions[-1]
        nh = attentions_all.shape[1]

        # Calculate feature map dimensions using crop_size and patch size
        w_featmap = crop_size[0] // model.config.patch_size
        h_featmap = crop_size[1] // model.config.patch_size
        num_patch_tokens = w_featmap * h_featmap

        num_register_tokens = model.config.num_register_tokens if hasattr(model.config, "num_register_tokens") else 0

        # Extract patch attention for the CLS token:
        # The attention matrix shape: [1, nh, seq_len, seq_len] where seq_len = 1 (CLS) + num_patch_tokens + extra_tokens.
        # We slice tokens from index 1 to 1+num_patch_tokens (ignoring CLS and extra tokens).
        attentions = attentions_all[0, :, 0,
                     (1 + num_register_tokens):(1 + num_register_tokens + num_patch_tokens)].reshape(nh,
                                                                                                     num_patch_tokens)

        # Aggregate all heads into one map by summing across the head dimension.
        combined_attention = torch.sum(attentions, dim=0)  # shape: (num_patch_tokens)
        # Normalize the combined attention map
        combined_attention = combined_attention / combined_attention.max()

        # Reshape combined attention to (w_featmap, h_featmap)
        combined_attention = combined_attention.reshape(w_featmap, h_featmap)

        # Interpolate combined attention map to original image size
        combined_attention = nn.functional.interpolate(
            combined_attention.unsqueeze(0).unsqueeze(0),
            size=size,
            mode="nearest"
        )[0, 0].cpu().numpy()

        pil_img_resized = pil_img.resize(size)
        original_img = np.array(pil_img_resized, dtype=np.float32) / 255.0

        # Overlay the combined attention map on the original image
        cam_image = show_cam_on_image(original_img, combined_attention, use_rgb=True)

        cv2.imwrite(f"{heatmap_dir}/{Path(img_path).name}", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{cam_dir}/{Path(img_path).stem}.png", (combined_attention * 255).astype(np.uint8))


if __name__ == '__main__':
    dataset_path = Path('/mnt/e/Datasets', 'SUN-SEG')
    enabled_datasets = [Path(dataset_path, 'TestEasyDataset', 'Seen'),
                        Path(dataset_path, 'TestEasyDataset', 'Unseen'),
                        Path(dataset_path, 'TestHardDataset', 'Seen'),
                        Path(dataset_path, 'TestHardDataset', 'Unseen'),
                        Path(dataset_path, 'TrainDataset')]

    frames = []
    for dataset in enabled_datasets:
        for case in (dataset / 'Frame').iterdir():
            for frame in (f for f in case.iterdir() if f.is_file() and f.name.endswith('.jpg')):
                frames.append({'image': str(frame), 'label': 1, 'image_path': str(frame)})

    use_dino(frames, Path('cams/dinov2'))
    use_densenet(frames, Path('cams/densenet121'))


