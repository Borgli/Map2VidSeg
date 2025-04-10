# Combined Pipeline: CAM Generation -> YOLOE Tracking -> SAM2 Refinement
# Generates CAMs on the fly using a classifier model.

import torch
import torch.nn as nn
import numpy as np
import cv2
import supervision as sv
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy.ndimage import binary_fill_holes
import os
import shutil
import torchvision.transforms.functional as TF
import torchvision.models as models # To load DenseNet
from torchvision import transforms

# --- Grad-CAM ---
# pip install grad-cam
try:
    from gradcam.utils.model_targets import ClassifierOutputTarget
    from gradcam.utils.image import show_cam_on_image # For visualization if needed
    from gradcam import GradCAM # Example CAM method
except ImportError:
    print("Error: pytorch-grad-cam not found. Please install it: pip install grad-cam")
    exit()

# Ultralytics / YOLOE specific imports
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
from ultralytics.engine.results import Results as UltralyticsResults
from ultralytics.trackers import register_tracker # Correct import for register_tracker

# SAM2 specific imports
from sam2.sam2_video_predictor import SAM2VideoPredictor

# --- Configuration ---

# Paths
BASE_DATA_DIR = Path('/mnt/e/Datasets/SUN-SEG')
BASE_EXPERIMENT_DIR = Path('/mnt/e/SAMexperiments/sam_video/cam_tests')

# Datasets to process (relative to BASE_DATA_DIR)
DATASET_DIRS = [
    BASE_DATA_DIR / 'TestEasyDataset' / 'Unseen',
    BASE_DATA_DIR / 'TestHardDataset' / 'Unseen',
]

# Experiment Name and Output Directory
CAM_MODEL_NAME = 'densenet121_generated' # Indicate CAMs are generated
EXPERIMENT_NAME = f'combined_pipeline_{CAM_MODEL_NAME}_yoloe_sam2'
OUTPUT_DIR = BASE_EXPERIMENT_DIR / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Checkpoints / Settings
# Classifier for CAM Generation (Example: DenseNet-121)
# NOTE: Replace with your actual trained classifier if needed.
# If using a standard torchvision model, ensure its output suits your task.
CLASSIFIER_MODEL_NAME = 'densenet121'
# Specify the target layer for CAMs (must exist in the loaded classifier)
# Common target for DenseNet-121 is the last convolutional layer in the final dense block
CLASSIFIER_TARGET_LAYER_NAME = 'features.denseblock4.denselayer16.conv2' # Adjust if needed!
# Specify the target class index for CAM generation (e.g., 0 if class 0 is 'polyp')
# Set to None if using a model with a single output (sigmoid) or want GradCAM to maximize output
CAM_TARGET_CLASS_INDEX = 0 # Adjust based on your classifier's output classes

YOLOE_CHECKPOINT = 'yoloe-11l-seg.pt' # Assumes download or path
SAM2_CHECKPOINT = 'facebook/sam2.1-hiera-large' # Hugging Face model ID

# Parameters
IMAGE_SIZE = 224 # Resize size for YOLOE/SAM, CAM might use different internal size
CLASSIFIER_INPUT_SIZE = 224 # Input size expected by the DenseNet classifier
YOLOE_CONF_THRESH = 0.25
YOLOE_IOU_THRESH = 0.7
TRACK_MIN_FRAMES = 3
GAP_FILL_ITERATIONS = 5
SAM2_PROMPT_INTERVAL = 10

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    TORCH_DTYPE = torch.bfloat16
    torch.autocast("cuda", dtype=TORCH_DTYPE).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        print("Activating TF32 for faster matmuls on Ampere+ GPUs")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    TORCH_DTYPE = torch.float32

# --- Model Loading ---

def load_actual_classifier(model_name='densenet121', num_classes=1) -> nn.Module:
    """Loads the actual classifier model for CAM generation."""
    print(f"Loading Classifier for CAM: {model_name}")
    if model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # *** IMPORTANT: Modify the final layer for your specific task ***
        # Example: Replace the classifier for binary classification (polyp vs no polyp)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        # Load your fine-tuned weights here if applicable
        # model.load_state_dict(torch.load("path/to/your/densenet_polyp_weights.pth"))
        print(f"Loaded DenseNet-121. Final layer replaced for {num_classes} output(s).")
        print("NOTE: Ensure you load your task-specific weights if fine-tuned!")
    # Add elif blocks here for other models like DINOv2 etc.
    # elif model_name == 'dinov2':
    #    from transformers import AutoModel
    #    model = AutoModel.from_pretrained(...)
    #    # Add classification head
    else:
        raise ValueError(f"Unsupported classifier model name: {model_name}")

    model.eval()
    model.to(DEVICE)
    return model

# Load the actual classifier
# Adjust num_classes based on your model's output (e.g., 1 for sigmoid binary, 2 for softmax binary)
classifier_model = load_actual_classifier(CLASSIFIER_MODEL_NAME, num_classes=1)

# Load YOLOE
print(f"Loading YOLOE model from: {YOLOE_CHECKPOINT}")
yoloe_model = YOLOE(model=YOLOE_CHECKPOINT)
register_tracker(yoloe_model, persist=True)
yoloe_model.eval()
yoloe_model.to(DEVICE)
print("YOLOE model loaded.")

# Load SAM2
print(f"Loading SAM2 Video Predictor from: {SAM2_CHECKPOINT}")
sam_predictor = SAM2VideoPredictor.from_pretrained(SAM2_CHECKPOINT)
sam_predictor.predictor.model.to(DEVICE)
print("SAM2 Video Predictor loaded.")

# --- Grad-CAM Setup ---
try:
    # Find the target layer module in the loaded model
    target_layer_module = None
    current_module = classifier_model
    for layer_name in CLASSIFIER_TARGET_LAYER_NAME.split('.'):
        current_module = getattr(current_module, layer_name)
    target_layer_module = current_module
    print(f"Identified Target Layer for CAM: {CLASSIFIER_TARGET_LAYER_NAME}")

    cam_executor = GradCAM(model=classifier_model, target_layers=[target_layer_module])
    print(f"Initialized GradCAM using target layer.")

except Exception as e:
    print(f"\n!!! Error setting up GradCAM: {e} !!!")
    print(f"Failed to find or use target layer: {CLASSIFIER_TARGET_LAYER_NAME}")
    print("Please verify the CLASSIFIER_TARGET_LAYER_NAME matches the classifier architecture.")
    print("You can inspect the model structure using print(classifier_model)")
    exit()


# --- Helper Functions ---

# Preprocessing for the classifier model (adjust mean/std if needed)
classifier_preprocess = transforms.Compose([
    transforms.Resize((CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image_for_classifier(image: Image.Image) -> torch.Tensor:
    """Preprocesses a PIL image for the classifier."""
    tensor = classifier_preprocess(image).unsqueeze(0) # Add batch dimension
    return tensor.to(DEVICE)

@torch.no_grad() # CAM generation should not require gradients here
def generate_cam(cam_tool: GradCAM, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray | None:
    """Generates a CAM heatmap for the input tensor."""
    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
    try:
        # The actual CAM computation might require gradients internally,
        # but we run this function wrapper without tracking gradients for our part.
        # Re-enable gradients temporarily just for the CAM call if needed by the lib
        with torch.enable_grad():
             grayscale_cam = cam_tool(input_tensor=input_tensor, targets=targets)

        # Take the first image in the batch and the first CAM result
        cam_image = grayscale_cam[0, :]

        # Resize CAM to the standard IMAGE_SIZE for consistency
        cam_image_resized = cv2.resize(cam_image, (IMAGE_SIZE, IMAGE_SIZE))

        # Normalize to 0-255 range
        cam_image_normalized = cv2.normalize(cam_image_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cam_image_normalized
    except Exception as e:
        print(f"Warning: CAM generation failed: {e}")
        return None

# Modified function to accept heatmap directly
def extract_initial_detections_from_cam(cam_heatmap: np.ndarray | None, image_shape: tuple) -> sv.Detections:
    """
    Extracts initial bounding boxes and masks from a generated CAM heatmap using thresholding
    and connected components.
    """
    if cam_heatmap is None:
        #print(f"Warning: Received None heatmap. Returning empty detections.")
        return sv.Detections.empty()

    # Ensure heatmap is uint8 grayscale
    if cam_heatmap.dtype != np.uint8:
         cam_heatmap = cv2.normalize(cam_heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # Use Otsu's thresholding to binarize the CAM
    # Apply a small Gaussian blur before Otsu can sometimes help
    # blurred_heatmap = cv2.GaussianBlur(cam_heatmap, (5, 5), 0)
    # _, binary_cam = cv2.threshold(blurred_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value, binary_cam = cv2.threshold(cam_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f"Otsu threshold: {threshold_value}") # Debugging

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cam)

    if num_labels <= 1:
        return sv.Detections.empty()

    bounding_boxes = []
    masks = []
    confidences = []

    # Normalize original heatmap to 0-1 for confidence calculation
    normalized_cam_for_conf = cam_heatmap / 255.0

    for i in range(1, num_labels): # Skip background
        x, y, w, h, area = stats[i]
        if area < 10: continue # Filter tiny components

        x1, y1, x2, y2 = x, y, x + w, y + h
        bounding_boxes.append([x1, y1, x2, y2])

        component_mask = (labels == i)
        masks.append(component_mask)

        # Calculate confidence based on the average *original* heatmap value within the component mask
        region_pixels = normalized_cam_for_conf[component_mask]
        region_confidence = np.mean(region_pixels) if region_pixels.size > 0 else 0.0
        confidences.append(region_confidence)

    if not bounding_boxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(bounding_boxes, dtype=np.float32),
        mask=np.array(masks, dtype=bool),
        confidence=np.array(confidences, dtype=np.float32)
    )

# --- [Rest of the helper functions remain largely the same] ---
# run_yoloe_track_with_prompts, filter_short_tracks, iou_numpy, save_final_results
# ... (Include the definitions from the previous response here) ...
@smart_inference_mode()
def run_yoloe_track_with_prompts(model: YOLOE, image: Image.Image, prompts: dict) -> UltralyticsResults:
    """Runs YOLOE tracking with specified visual prompts (bboxes or masks)."""
    results = model.track(
        source=np.array(image),       # PIL Image
        imgsz=IMAGE_SIZE,
        conf=YOLOE_CONF_THRESH,
        iou=YOLOE_IOU_THRESH,
        prompts=prompts,
        predictor=YOLOEVPSegPredictor, # Use the visual prompt predictor
        persist=True,      # Persist tracks across frames
        verbose=False      # Suppress individual frame output
    )
    return results[0]

def filter_short_tracks(results_list: list, min_frames: int) -> list:
    """Filters YOLOE results to remove tracks appearing in fewer than min_frames."""
    print(f"Filtering tracks: Keeping only those present for >= {min_frames} frames.")
    id_counter = Counter()
    for result, _ in results_list:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy()
            id_counter.update(ids)

    filtered_results_list = []
    for result, frame_path in results_list:
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy()
            valid_mask = [id_counter[id] >= min_frames for id in ids]
            if len(valid_mask) > 0:
                valid_mask_tensor = torch.tensor(valid_mask, device=result.boxes.id.device)
                result.boxes = result.boxes[valid_mask_tensor]
                if result.masks is not None:
                    result.masks = result.masks[valid_mask_tensor]
                    if result.masks.shape[0] == 0: result.masks = None
                if result.boxes.shape[0] == 0: result.boxes = None; result.masks = None
            else: result.boxes = None; result.masks = None
        else: result.boxes = None; result.masks = None
        filtered_results_list.append((result, frame_path))
    print("Filtering complete.")
    return filtered_results_list

def iou_numpy(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculates IoU between two boolean masks."""
    if mask1 is None or mask2 is None: return 0.0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-6)

# Supervision annotators (initialized globally for convenience)
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.5)
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_position=sv.Position.CENTER)

def save_final_results(results_list: list, output_dir: Path, case_name: str):
    """Saves the final masks and annotated plots."""
    plot_dir = output_dir / 'plots_final' / case_name
    mask_dir = output_dir / 'masks_final' / case_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving final results for case: {case_name}")
    for result_data, frame_path in tqdm(results_list, desc="Saving final outputs", leave=False):
        frame_stem = frame_path.stem
        try:
            frame_img_bgr = cv2.imread(str(frame_path))
            if frame_img_bgr is None: raise IOError("Could not read frame")
            frame_img_bgr = cv2.resize(frame_img_bgr, (IMAGE_SIZE, IMAGE_SIZE))
            frame_img_rgb = cv2.cvtColor(frame_img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not read/process frame {frame_path} for saving plot: {e}")
            continue

        combined_mask_binary = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        detections = sv.Detections.empty()
        annotated_frame = frame_img_rgb.copy()

        try:
            if isinstance(result_data, UltralyticsResults):
                if result_data.masks is not None and len(result_data.masks) > 0:
                    masks_np = result_data.masks.data.cpu().numpy().astype(bool)
                    combined_mask_binary = np.any(masks_np, axis=0)
                    detections = sv.Detections.from_ultralytics(result_data)
            elif isinstance(result_data, dict): # SAM2 results
                if 1 in result_data and result_data[1] is not None:
                    mask_obj = result_data[1]
                    if mask_obj.ndim == 3: mask_obj = mask_obj[0]
                    combined_mask_binary = mask_obj.astype(bool)
                    if combined_mask_binary.sum() > 0: # Only create detection if mask is not empty
                        all_masks = [combined_mask_binary]
                        all_boxes = [sv.mask_to_xyxy(np.array([combined_mask_binary]))[0]]
                        detections = sv.Detections(xyxy=np.array(all_boxes), mask=np.array(all_masks))
            # Annotate if detections were created
            if detections.xyxy.shape[0] > 0:
                annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        except Exception as e:
            print(f"Warning: Error processing result data for frame {frame_stem}: {e}")
            # Keep blank mask/frame

        # Save mask and plot
        combined_mask_save = combined_mask_binary.astype(np.uint8) * 255
        mask_save_path = (mask_dir / frame_stem).with_suffix('.png')
        plot_save_path = plot_dir / f"{frame_stem}.jpg"
        cv2.imwrite(str(mask_save_path), combined_mask_save)
        cv2.imwrite(str(plot_save_path), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))


# --- Main Processing Logic ---

if __name__ == '__main__':
    start_time_total = torch.cuda.Event(enable_timing=True)
    end_time_total = torch.cuda.Event(enable_timing=True)
    start_time_total.record()

    # Create top-level output directories
    cam_output_dir = OUTPUT_DIR / 'generated_cams' # Save generated CAMs here
    yoloe_output_dir = OUTPUT_DIR / 'yoloe_intermediate'
    sam2_output_dir = OUTPUT_DIR # Final output goes here
    cam_output_dir.mkdir(parents=True, exist_ok=True)
    yoloe_output_dir.mkdir(parents=True, exist_ok=True)
    sam2_output_dir.mkdir(parents=True, exist_ok=True)


    for dataset_dir in DATASET_DIRS:
        dataset_name = f"{dataset_dir.parent.name}_{dataset_dir.name}"
        print(f"\n--- Processing Dataset: {dataset_name} ---")

        frame_root_dir = dataset_dir / 'Frame'
        if not frame_root_dir.is_dir():
            print(f"Warning: Frame directory not found for {dataset_name}, skipping: {frame_root_dir}")
            continue

        # Output directories for this dataset
        dataset_cam_out = cam_output_dir / dataset_name
        dataset_yoloe_out = yoloe_output_dir / dataset_name
        dataset_sam2_out = sam2_output_dir / dataset_name
        dataset_cam_out.mkdir(parents=True, exist_ok=True)
        dataset_yoloe_out.mkdir(parents=True, exist_ok=True)
        dataset_sam2_out.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Generate CAMs and Extract Initial Detections ---
        all_initial_annotations = {} # key: frame_stem, value: sv.Detections
        print("Stage 1: Generating CAMs & Extracting initial detections...")
        case_dirs = sorted([d for d in frame_root_dir.iterdir() if d.is_dir()])

        if not case_dirs:
             print(f"Warning: No case subdirectories found in {frame_root_dir}. Skipping dataset.")
             continue

        for case_dir in tqdm(case_dirs, desc=f"Generating CAMs for {dataset_name}"):
            case_name = case_dir.name
            frame_paths = sorted(list(case_dir.glob('*.jpg')))
            cam_case_out_dir = dataset_cam_out / case_name
            cam_case_out_dir.mkdir(parents=True, exist_ok=True) # Dir to save generated CAMs

            if not frame_paths:
                 print(f"Warning: No frames found in case {case_name}. Skipping.")
                 continue

            for frame_path in frame_paths:
                frame_stem = frame_path.stem
                try:
                    # 1. Load Image
                    image_pil = Image.open(frame_path).convert("RGB")

                    # 2. Preprocess for Classifier
                    input_tensor = preprocess_image_for_classifier(image_pil)

                    # 3. Generate CAM
                    generated_heatmap = generate_cam(cam_executor, input_tensor, CAM_TARGET_CLASS_INDEX)

                    # 4. Save the generated CAM (optional, for debugging/visualization)
                    if generated_heatmap is not None:
                        cam_save_path = (cam_case_out_dir / frame_stem).with_suffix('.png')
                        cv2.imwrite(str(cam_save_path), generated_heatmap)
                    else:
                        # Create a blank image if CAM fails? Or handle downstream?
                         generated_heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) # Blank if failed


                    # 5. Extract Detections from generated CAM
                    initial_detections = extract_initial_detections_from_cam(generated_heatmap, (IMAGE_SIZE, IMAGE_SIZE))
                    all_initial_annotations[frame_stem] = initial_detections

                except Exception as e:
                    print(f"Error processing frame {frame_path} in Stage 1: {e}")
                    all_initial_annotations[frame_stem] = sv.Detections.empty()


        # --- Stage 2: YOLOE Multi-Pass Tracking & Refinement ---
        # --- [This Stage remains identical to the previous version] ---
        # It now uses all_initial_annotations populated by the generated CAMs.
        # ... (Paste Stage 2 code block here from previous response) ...
        print("\nStage 2: Running YOLOE Multi-Pass Tracking...")
        for case_dir in tqdm(case_dirs, desc=f"Processing cases in {dataset_name}"):
            case_name = case_dir.name
            frame_paths = sorted(list(case_dir.glob('*.jpg')))

            if not frame_paths: continue

            yoloe_case_out_dir = dataset_yoloe_out / case_name
            yoloe_case_out_dir.mkdir(parents=True, exist_ok=True)

            # --- YOLOE Pass 1: Prompting with Initial BBoxes ---
            print(f"  Case {case_name}: YOLOE Pass 1 (BBox Prompts)")
            results_list_pass1 = []
            yoloe_model.predictor.reset_tracker()
            for frame_path in tqdm(frame_paths, desc="Pass 1", leave=False):
                frame_stem = frame_path.stem
                image = Image.open(frame_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                initial_detections = all_initial_annotations.get(frame_stem, sv.Detections.empty())
                if initial_detections.xyxy.shape[0] > 0:
                    prompts = {
                        "bboxes": np.array(initial_detections.xyxy, dtype=np.float32),
                        "cls": np.zeros(len(initial_detections.xyxy), dtype=np.int32)
                    }
                    result = run_yoloe_track_with_prompts(yoloe_model, image, prompts)
                else:
                    result = yoloe_model.track(source=np.array(image), imgsz=IMAGE_SIZE, persist=True, verbose=False)[0]
                results_list_pass1.append((result, frame_path))

            # --- YOLOE Filtering: Remove Short Tracks ---
            print(f"  Case {case_name}: Filtering short tracks (< {TRACK_MIN_FRAMES} frames)")
            results_list_filtered = filter_short_tracks(results_list_pass1, TRACK_MIN_FRAMES)

            # --- YOLOE Pass 2: Prompting with Initial Masks for Missing Frames ---
            print(f"  Case {case_name}: YOLOE Pass 2 (Initial Mask Prompts for Gaps)")
            results_list_pass2 = []
            yoloe_model.predictor.reset_tracker()
            for result, frame_path in tqdm(results_list_filtered, desc="Pass 2", leave=False):
                if result.masks is None or len(result.masks) == 0:
                    frame_stem = frame_path.stem
                    initial_detections = all_initial_annotations.get(frame_stem, sv.Detections.empty())
                    if initial_detections.mask is not None and initial_detections.mask.shape[0] > 0:
                        image = Image.open(frame_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                        initial_masks_filled = binary_fill_holes(initial_detections.mask).astype(np.uint8)
                        prompts = { "masks": initial_masks_filled, "cls": np.zeros(len(initial_masks_filled), dtype=np.int32) }
                        result_pass2 = run_yoloe_track_with_prompts(yoloe_model, image, prompts)
                        results_list_pass2.append((result_pass2, frame_path))
                    else:
                        results_list_pass2.append((result, frame_path))
                else:
                    results_list_pass2.append((result, frame_path))

            # --- YOLOE Pass 3: Iterative Gap Filling (Forward) ---
            print(f"  Case {case_name}: YOLOE Pass 3 (Iterative Forward Gap Filling)")
            results_list_pass3f = list(results_list_pass2)
            for iteration in range(GAP_FILL_ITERATIONS):
                new_masks_generated = False
                print(f"    Iteration {iteration + 1}/{GAP_FILL_ITERATIONS}")
                yoloe_model.predictor.reset_tracker()
                temp_results = []
                for index, (result, frame_path) in tqdm(enumerate(results_list_pass3f), desc=f"Fwd Iter {iteration+1}", leave=False, total=len(results_list_pass3f)):
                    if (result.masks is None or len(result.masks) == 0) and index > 0:
                        prev_result, _ = results_list_pass3f[index - 1]
                        if prev_result.masks is not None and len(prev_result.masks) > 0:
                            image = Image.open(frame_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                            prev_masks = prev_result.masks.data.cpu().numpy().astype(np.uint8)
                            prompts = { "masks": prev_masks, "cls": np.zeros(len(prev_masks), dtype=np.int32) }
                            result_filled = run_yoloe_track_with_prompts(yoloe_model, image, prompts)
                            if result_filled.masks is not None and len(result_filled.masks) > 0: new_masks_generated = True
                            temp_results.append((result_filled, frame_path))
                        else: temp_results.append((result, frame_path))
                    else: temp_results.append((result, frame_path))
                results_list_pass3f = temp_results
                if not new_masks_generated: print("    No new masks generated in forward pass, stopping early."); break

            # --- YOLOE Pass 4: Iterative Gap Filling (Backward) ---
            print(f"  Case {case_name}: YOLOE Pass 4 (Iterative Backward Gap Filling)")
            results_list_pass4b = list(results_list_pass3f); results_list_pass4b.reverse()
            for iteration in range(GAP_FILL_ITERATIONS):
                new_masks_generated = False
                print(f"    Iteration {iteration + 1}/{GAP_FILL_ITERATIONS}")
                yoloe_model.predictor.reset_tracker()
                temp_results_reversed = []
                for index, (result, frame_path) in tqdm(enumerate(results_list_pass4b), desc=f"Bwd Iter {iteration+1}", leave=False, total=len(results_list_pass4b)):
                    if (result.masks is None or len(result.masks) == 0) and index > 0:
                        next_result, _ = results_list_pass4b[index - 1]
                        if next_result.masks is not None and len(next_result.masks) > 0:
                            image = Image.open(frame_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                            next_masks = next_result.masks.data.cpu().numpy().astype(np.uint8)
                            prompts = { "masks": next_masks, "cls": np.zeros(len(next_masks), dtype=np.int32) }
                            result_filled = run_yoloe_track_with_prompts(yoloe_model, image, prompts)
                            if result_filled.masks is not None and len(result_filled.masks) > 0: new_masks_generated = True
                            temp_results_reversed.append((result_filled, frame_path))
                        else: temp_results_reversed.append((result, frame_path))
                    else: temp_results_reversed.append((result, frame_path))
                results_list_pass4b = temp_results_reversed
                if not new_masks_generated: print("    No new masks generated in backward pass, stopping early."); break
            results_list_pass4b.reverse(); yoloe_results_final = results_list_pass4b

            # --- YOLOE Fallback: Use Initial CAM Mask if Still No Mask ---
            print(f"  Case {case_name}: YOLOE Fallback (Using Initial CAM for remaining gaps)")
            yoloe_results_final_fallback = []
            for result, frame_path in yoloe_results_final:
                 if result.masks is None or len(result.masks) == 0:
                    frame_stem = frame_path.stem
                    initial_detections = all_initial_annotations.get(frame_stem, sv.Detections.empty())
                    if initial_detections.mask is not None and initial_detections.mask.shape[0] > 0:
                        masks_fallback = torch.from_numpy(initial_detections.mask).unsqueeze(1).to(DEVICE)
                        boxes_fallback = torch.tensor([[0,0,IMAGE_SIZE, IMAGE_SIZE, 0.1, 0]]*len(masks_fallback), device=DEVICE)
                        fallback_result = UltralyticsResults(
                             orig_img=cv2.cvtColor(np.array(Image.open(frame_path).resize((IMAGE_SIZE, IMAGE_SIZE))), cv2.COLOR_RGB2BGR), # Need an image array
                             path=str(frame_path), names=yoloe_model.model.names,
                             boxes=boxes_fallback, masks=masks_fallback)
                        yoloe_results_final_fallback.append((fallback_result, frame_path))
                    else: yoloe_results_final_fallback.append((result, frame_path))
                 else: yoloe_results_final_fallback.append((result, frame_path))


        # --- Stage 3: SAM2 Double-Pass Refinement ---
        # --- [This Stage remains identical to the previous version] ---
        # It uses yoloe_results_final_fallback populated by the YOLOE stage.
        # ... (Paste Stage 3 code block here from previous response) ...
            print(f"\nStage 3: Running SAM2 Double-Pass Refinement for case {case_name}...")
            sam_frame_dir = dataset_sam2_out / 'temp_sam_frames' / case_name
            sam_frame_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Preparing frames for SAM2 in {sam_frame_dir}")
            for frame_idx, (_, frame_path) in enumerate(yoloe_results_final_fallback):
                temp_frame_path = sam_frame_dir / f'{frame_idx:05d}.jpg'
                if not temp_frame_path.exists():
                    Image.open(frame_path).resize((IMAGE_SIZE, IMAGE_SIZE)).save(temp_frame_path)

            sam_masks_for_prompting = []
            for result, _ in yoloe_results_final_fallback:
                if isinstance(result, UltralyticsResults) and result.masks is not None and len(result.masks) > 0:
                     combined_mask = np.any(result.masks.data.cpu().numpy(), axis=0).astype(bool)
                     sam_masks_for_prompting.append(combined_mask)
                else: sam_masks_for_prompting.append(None)

            # --- SAM2 Forward Pass ---
            print("  Running SAM2 Forward Pass...")
            video_segments_forward = {}
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=TORCH_DTYPE):
                    inference_state = sam_predictor.init_state(video_path=sam_frame_dir)
                    for frame_idx, mask_to_add in tqdm(enumerate(sam_masks_for_prompting), total=len(sam_masks_for_prompting), desc="Adding Fwd Prompts", leave=False):
                        if frame_idx % SAM2_PROMPT_INTERVAL == 0 and mask_to_add is not None:
                            sam_predictor.add_new_mask(inference_state=inference_state, frame_idx=frame_idx, obj_id=1, mask=mask_to_add)
                    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(sam_predictor.propagate_in_video(inference_state), desc="Propagating Fwd", leave=False):
                        obj_index = out_obj_ids.index(1) if 1 in out_obj_ids else -1
                        if obj_index != -1:
                             mask = (out_mask_logits[obj_index] > 0.0).cpu().numpy()
                             video_segments_forward[out_frame_idx] = {1: mask}
            except Exception as e: print(f"Error during SAM2 Forward Pass for {case_name}: {e}")

            # --- SAM2 Backward Pass ---
            print("  Running SAM2 Backward Pass...")
            video_segments_reverse = {}
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=TORCH_DTYPE):
                    sam_predictor.reset_state(inference_state)
                    prompt_indices = range(len(sam_masks_for_prompting) - 1, -1, -1)
                    for frame_idx in tqdm(prompt_indices, total=len(sam_masks_for_prompting), desc="Adding Bwd Prompts", leave=False):
                         if frame_idx % SAM2_PROMPT_INTERVAL == 0:
                            mask_to_add = video_segments_forward.get(frame_idx, {}).get(1, sam_masks_for_prompting[frame_idx])
                            if mask_to_add is not None:
                                sam_predictor.add_new_mask(inference_state=inference_state, frame_idx=frame_idx, obj_id=1, mask=mask_to_add)
                    last_frame_idx = len(sam_masks_for_prompting) - 1
                    for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(sam_predictor.propagate_in_video(inference_state, start_frame_idx=last_frame_idx, reverse=True), desc="Propagating Bwd", leave=False):
                         obj_index = out_obj_ids.index(1) if 1 in out_obj_ids else -1
                         if obj_index != -1:
                             mask = (out_mask_logits[obj_index] > 0.0).cpu().numpy()
                             video_segments_reverse[out_frame_idx] = {1: mask}
            except Exception as e: print(f"Error during SAM2 Backward Pass for {case_name}: {e}")

            # --- SAM2 Merging Pass ---
            print("  Merging SAM2 Forward and Backward Passes...")
            video_segments_final = {}
            all_frame_indices = sorted(list(set(video_segments_forward.keys()) | set(video_segments_reverse.keys())))
            for i, frame_idx in enumerate(all_frame_indices):
                mask_fwd = video_segments_forward.get(frame_idx, {}).get(1, None)
                mask_bwd = video_segments_reverse.get(frame_idx, {}).get(1, None)
                if mask_fwd is not None and mask_bwd is None: best_mask = mask_fwd
                elif mask_fwd is None and mask_bwd is not None: best_mask = mask_bwd
                elif mask_fwd is None and mask_bwd is None: best_mask = None
                else:
                    prev_idx = all_frame_indices[i - 1] if i > 0 else -1
                    prev_mask_final = video_segments_final.get(prev_idx, {}).get(1, None)
                    iou_fwd_prev = iou_numpy(mask_fwd, prev_mask_final)
                    iou_bwd_prev = iou_numpy(mask_bwd, prev_mask_final)
                    best_mask = mask_fwd if iou_fwd_prev >= iou_bwd_prev else mask_bwd
                if best_mask is not None: video_segments_final[frame_idx] = {1: best_mask}

            # --- Prepare Final Output List ---
            final_results_list = []
            original_frame_paths_map = {i: fp for i, (_, fp) in enumerate(yoloe_results_final_fallback)}
            for frame_idx in range(len(original_frame_paths_map)):
                 frame_path = original_frame_paths_map[frame_idx]
                 sam_result_data = video_segments_final.get(frame_idx, {})
                 final_results_list.append((sam_result_data, frame_path))

            # --- Save Final SAM2 Results ---
            save_final_results(final_results_list, dataset_sam2_out, case_name)

            # Clean up temporary SAM frames
            print(f"  Cleaning up temporary frames: {sam_frame_dir}")
            shutil.rmtree(sam_frame_dir)

            print(f"--- Finished processing case: {case_name} ---")

    end_time_total.record()
    torch.cuda.synchronize()
    total_time = start_time_total.elapsed_time(end_time_total) / 1000
    print(f"\n=== Pipeline Finished ===")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Final outputs saved in: {sam2_output_dir}")
    print(f"Generated CAMs saved in: {cam_output_dir}")