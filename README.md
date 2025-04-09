# Map2VidSeg

This is the repository for the paper ""


## Installation
We have only tested the code on WSL2 with Ubuntu 24.04 and Python 3.12.3 with CUDA 12.4.

For DINOv2 and YOLOE we use Hugging Face and Ultralytics and dependencies can be installed with:
```bash
pip install -r requirements.txt
```

We can not use the Ultralytics version of SAM-2 as it does not support adding boxes to specific frames which we need
for tracing objects.
For SAM-2, see the SAM-2 installation in the [official SAM-2 repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation).

## Datasets
For testing the model, use the [SUN-SEG annotated dataset](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md).
We used [TestEasyDataset Unseen](https://paperswithcode.com/dataset/sun-seg-easy) and [TestHardDataset Unseen](https://paperswithcode.com/dataset/sun-seg-hard) for evaluating our results.

For training the classifier, we supply the [pretrained weights](https://huggingface.co/borgli/dinov2-polyp-classifier) for the DINOv2 model
and [pretrained weights](https://huggingface.co/borgli/densenet121-polyp-classifier) for the DenseNet-121 model. However, if you want to use a different classifier, or want to recreate our experiments, follow the instructions in [DATA_PREPARATION.md](DATA_PREPARATION.md).


## Using the Pipeline
The pipeline is compatible with huggingface models or the code can be changed to custom models,
but a classifier is required, and it needs to be either a CNN model compatible with the PyTorch CAM library or
a VIT model where we can extract the attention maps. For CNN models we use the final feature layer for the CAM and
for VIT we use the final block added together.


## Recreating the experiments
