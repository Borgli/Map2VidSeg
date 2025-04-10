from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import cv2
import supervision as sv

import json
import shutil
from collections import defaultdict
from scipy.ndimage import binary_fill_holes
import cv2
import numpy as np
from supervision import Detections, box_iou_batch
from tqdm import tqdm
import torch
from collections import Counter
import re

from ultralytics import YOLOE, SAM, RTDETR, YOLO
from pathlib import Path
from PIL import Image
from ultralytics.utils.torch_utils import smart_inference_mode
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import supervision as sv
from ultralytics.engine.results import Masks, Boxes

from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState

from yoloe.ultralytics.data.loaders import LoadImagesAndVideos
from yoloe.ultralytics.utils import autobatch

from tracking import process_frame, merge_tracks, iou
from yoloe.ultralytics.trackers import register_tracker

#test_dir = Path('/mnt/e/SAMexperiments/test_video/test_12_sec')

test_dirs = [Path('/mnt/e/Datasets/SUN-SEG/TestEasyDataset/Unseen/'),
             #Path('/mnt/e/Datasets/SUN-SEG/TestEasyDataset/Seen/'),
             Path('/mnt/e/Datasets/SUN-SEG/TestHardDataset/Unseen/'),]
             #Path('/mnt/e/Datasets/SUN-SEG/TestHardDataset/Seen/')]
cams_dir = Path('/mnt/e/SAMexperiments/sam_video/cams')

experiment_name = 'tracking_yoloe_redo'

out_dir = Path('/mnt/e/SAMexperiments/sam_video/cam_tests') / experiment_name
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
byte_track = sv.ByteTrack(frame_rate=20, track_activation_threshold=0.1, minimum_matching_threshold=0.2)
tracker_id = 0

def extract_bounding_boxes_dino(cams_dir, test_dir):
    frame_annotations = {}
    for index, (frame_path) in tqdm(enumerate(test_dir.iterdir()), total=len(list(test_dir.iterdir())), desc='Extracting bounding boxes', postfix='case: ' + test_dir.name):
        #frame = Image.open(frame_path).convert("RGB")

        cam_path = (cams_dir / 'cam' / frame_path.stem).with_suffix('.png')
        cam = cv2.imread(str(cam_path), cv2.IMREAD_GRAYSCALE)
        # normalize cam
        _, binary_cam = cv2.threshold(cam, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cam)

        #heatmap = cv2.imread(str((cams_dir / 'heatmap' / frame_path.name)), cv2.IMREAD_COLOR_RGB)

        bounding_boxes = np.array([[x, y, x + w, y + h] for x, y, w, h, _ in stats[1:]])
        masks = []
        confidences = []
        for i, (x, y, x2, y2) in enumerate(bounding_boxes, start=1):
            region = cam[y:y2, x:x2] / 255.0
            confidences.append(np.mean(region))
            full_mask = np.zeros_like(cam, dtype=bool)
            component_mask = (labels[y:y2, x:x2] == i)
            full_mask[y:y2, x:x2] = component_mask
            masks.append(full_mask)

        confidences = np.array(confidences)
        masks = np.array(masks)

        detections = sv.Detections(xyxy=bounding_boxes, mask=masks, confidence=confidences)
        #detections = sv.Detections(xyxy=bounding_boxes, class_id=np.ones(len(bounding_boxes)), confidence=np.array(confidences), tracker_id=np.array([increment_tracker_id() for _ in range(len(bounding_boxes))]))
        #detections = byte_track.update_with_detections(detections)

        #bounding_box = box_annotator.annotate(heatmap, detections)
        #cv2.imwrite(str(out_dir / frame_path.name), cv2.cvtColor(bounding_box, cv2.COLOR_RGB2BGR))

        frame_annotations[cam_path.stem] = detections

    return frame_annotations


def init_model(checkpoint='pretrain/yoloe-11l-seg.pt'):
    model = YOLOE(model=checkpoint)
    register_tracker(model, persist=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


@smart_inference_mode()
def infer_with_bboxes(image, bboxes, model, tracker, image_size=224, conf_thresh=0.25, iou_thresh=0.7):
    if bboxes.any():
        # Prepare prompt with bounding boxes (each box is [x1, y1, x2, y2])
        prompts = {
            "bboxes": np.array(bboxes, dtype=np.float32),
            "cls": np.array([0] * len(bboxes))  # Single class "polyp" as class 0
        }

        # Pass the visual prompt (bounding boxes) with the predictor
        results = model.track(
            source=image,
            imgsz=image_size,
            conf=conf_thresh,
            iou=iou_thresh,
            prompts=prompts,
            predictor=YOLOEVPSegPredictor,
            persist=True
        )


        if results[0].masks and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            masks = results[0].masks.data.cpu().numpy()
            track_info = zip(boxes, masks, track_ids)
        else:
            track_info = []
        #plot_image = results[0].plot()
        # Annotate detections
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
    else:
        #results = model.track(source=image, persist=True)
        #detections = sv.Detections.empty()
        plot_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        detections = Detections.empty()
        track_info = ()
    #else:
    #    detections = sv.Detections.empty()

    plot_image = None

    detections.xyxy = np.atleast_2d(detections.xyxy)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
    labels = [f"polyp {conf:.2f}" for conf in detections.confidence]
    annotated_image = image.copy()
    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4) \
        .annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness) \
        .annotate(scene=annotated_image, detections=detections)
    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale, smart_position=True) \
        .annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image, plot_image, track_info


if __name__ == '__main__':
    for test_dir in test_dirs:
        frame_annotations = {}
        out_dir_dataset = out_dir / f"{test_dir.parent.name}_{test_dir.name}"
        out_dir_dataset.mkdir(parents=True, exist_ok=True)

        test_dir = test_dir / 'Frame'
        for case in list(test_dir.iterdir()):
            if case.is_dir():
                frame_annotations.update(extract_bounding_boxes_dino(cams_dir / 'dinov2', case))


        image_size = 224
        conf_thresh = 0.25
        iou_thresh = 0.7

        tracker = sv.ByteTrack(frame_rate=20)

        model = init_model()

        #batch_size = autobatch.autobatch(model, 224, True, 0.6, 1)
        #print('Batch size:', batch_size)

        # First pass
        out_annotated = out_dir_dataset / 'first_pass'
        out_annotated.mkdir(parents=True, exist_ok=True)

        out_raw_mask = out_annotated / 'raw_mask'
        out_raw_mask.mkdir(parents=True, exist_ok=True)

        out_annotated_mask = out_annotated / 'annotated_mask'
        out_annotated_mask.mkdir(parents=True, exist_ok=True)

        out_annotated_prompt = out_annotated / 'annotated_prompt'
        out_annotated_prompt.mkdir(parents=True, exist_ok=True)


        # First pass should be simple. We use only bounding boxes.
        for case in list(test_dir.iterdir()):
            results_list = []
            if case.is_dir():
                for frame_path in tqdm(case.iterdir(), total=len(list(case.iterdir())),
                                       desc=f'Annotating frames in {test_dir.parent.parent.name}_{test_dir.parent.name}',
                                       postfix=f'case: {case.name}'):
                    frame = Image.open(frame_path).convert("RGB").resize((224, 224))
                    detections = frame_annotations[frame_path.stem]

                    bboxes = detections.xyxy

                    annotated_prompts = box_annotator.annotate(np.array(frame.copy()), detections)
                    cv2.imwrite(str(out_annotated_prompt / frame_path.name), cv2.cvtColor(annotated_prompts, cv2.COLOR_RGB2BGR))

                    prompts = {
                        "bboxes": np.array(bboxes, dtype=np.float32),
                        "cls": np.array([0] * len(bboxes))  # Single class "polyp" as class 0
                    }

                    # Pass the visual prompt (bounding boxes) with the predictor
                    results = model.track(
                        source=frame,
                        imgsz=image_size,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        prompts=prompts,
                        predictor=YOLOEVPSegPredictor,
                        persist=True
                    )

                    results_list.append((results[0], frame_path))

                    cv2.imwrite(str(out_annotated_mask / frame_path.name), results[0].plot())

                    #annotated, track_plot, track_info = infer_with_bboxes(frame, bboxes_list, model, tracker)
                    #annotated.save(str(out_annotated / f"{frame_path.stem}.jpg"))
                    #Image.fromarray(cv2.cvtColor(track_plot, cv2.COLOR_RGB2BGR)).save(
                    #    str(out_annotated_tracked / f"{frame_id}_track.jpg"))



            # Step 1: Count tracking IDs across all frames
            id_counter = Counter()
            for result, frame_path in results_list:
                # Move tracking IDs to CPU and convert to numpy
                if result.boxes.id is not None:
                    ids = result.boxes.id.cpu().numpy()
                    id_counter.update(ids)

            # Step 2: Filter detections for IDs that appear in less than 3 frames
            for result, frame_path in results_list:
                if result.boxes.id is not None:
                    # Get tracking IDs for this result
                    ids = result.boxes.id.cpu().numpy()
                    # Create a boolean list: True if ID count is >= 3, False otherwise
                    valid = [id_counter[id] >= 3 for id in ids]
                    # Convert the list to a torch tensor on the same device as the original IDs
                    valid_tensor = torch.tensor(valid, device=result.boxes.id.device)

                    # Option A: Remove the entire detection (both box and mask)
                    result.boxes = result.boxes[valid_tensor]
                    result.masks = result.masks[valid_tensor]

                    if result.masks.shape[0] == 0:
                        result.masks = None


            # Second pass
            out_annotated = out_dir_dataset / 'second_pass'
            out_annotated.mkdir(parents=True, exist_ok=True)

            out_raw_mask = out_annotated / 'raw_mask'
            out_raw_mask.mkdir(parents=True, exist_ok=True)

            out_annotated_mask = out_annotated / 'annotated_mask'
            out_annotated_mask.mkdir(parents=True, exist_ok=True)

            out_annotated_prompt = out_annotated / 'annotated_prompt'
            out_annotated_prompt.mkdir(parents=True, exist_ok=True)

            model = init_model()

            for result, frame_path in results_list:
                if result.masks is None:
                    frame = Image.open(frame_path).convert("RGB").resize((224, 224))
                    detections = frame_annotations[frame_path.stem]

                    bboxes = detections.xyxy
                    masks = detections.mask
                    #masks = binary_fill_holes(masks).astype(np.uint8)
                    #masks[masks > 0] = 1

                    annotated_prompts = box_annotator.annotate(np.array(frame.copy()), detections)
                    cv2.imwrite(str(out_annotated_prompt / frame_path.name), cv2.cvtColor(annotated_prompts, cv2.COLOR_RGB2BGR))

                    prompts = {
                        "masks": np.array(masks).astype(np.uint8),
                        "cls": np.array([0] * len(bboxes))  # Single class "polyp" as class 0
                    }

                    # Pass the visual prompt (bounding boxes) with the predictor
                    results = model.track(
                        source=np.array(frame),
                        imgsz=image_size,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        prompts=prompts,
                        predictor=YOLOEVPSegPredictor,
                        persist=True
                    )

                    cv2.imwrite(str(out_annotated_mask / frame_path.name), results[0].plot())



            iteration = 0
            new_masks_generated = True
            while new_masks_generated and iteration < 5:
                new_masks_generated = False
                # Third pass
                out_annotated = out_dir_dataset / f'third_pass_iteration_{iteration}'
                iteration += 1
                out_annotated.mkdir(parents=True, exist_ok=True)

                out_raw_mask = out_annotated / 'raw_mask'
                out_raw_mask.mkdir(parents=True, exist_ok=True)

                out_annotated_mask = out_annotated / 'annotated_mask'
                out_annotated_mask.mkdir(parents=True, exist_ok=True)

                out_annotated_prompt = out_annotated / 'annotated_prompt'
                out_annotated_prompt.mkdir(parents=True, exist_ok=True)

                model = init_model()

                for index, (result, frame_path) in enumerate(results_list):
                    if result.masks is None and results_list[index - 1][0].masks:
                        frame = Image.open(frame_path).convert("RGB").resize((224, 224))
                        detections = frame_annotations[frame_path.stem]

                        bboxes = detections.xyxy
                        #masks = detections.mask
                        # masks = binary_fill_holes(masks).astype(np.uint8)
                        # masks[masks > 0] = 1

                        if results_list[index - 1][0].masks:
                            masks = results_list[index - 1][0].masks.data.cpu().numpy().astype(np.uint8)

                        annotated_prompts = box_annotator.annotate(np.array(frame.copy()), detections)
                        cv2.imwrite(str(out_annotated_prompt / frame_path.name),
                                    cv2.cvtColor(annotated_prompts, cv2.COLOR_RGB2BGR))

                        prompts = {
                            "masks": masks,
                            "cls": np.array([0] * len(bboxes))  # Single class "polyp" as class 0
                        }

                        # Pass the visual prompt (bounding boxes) with the predictor
                        results = model.track(
                            source=frame,
                            imgsz=image_size,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            prompts=prompts,
                            predictor=YOLOEVPSegPredictor,
                            persist=True
                        )

                        if results[0].masks:
                            new_masks_generated = True

                        results_list[index] = (results[0], frame_path)

                        cv2.imwrite(str(out_annotated_mask / frame_path.name), results[0].plot())
                        print(result)

            results_list.reverse()
            new_masks_generated = True
            iteration = 0
            while new_masks_generated and iteration < 5:
                new_masks_generated = False
                # Third pass
                out_annotated = out_dir_dataset / f'third_pass_reversed_iteration_{iteration}'
                iteration += 1
                out_annotated.mkdir(parents=True, exist_ok=True)

                out_raw_mask = out_annotated / 'raw_mask'
                out_raw_mask.mkdir(parents=True, exist_ok=True)

                out_annotated_mask = out_annotated / 'annotated_mask'
                out_annotated_mask.mkdir(parents=True, exist_ok=True)

                out_annotated_prompt = out_annotated / 'annotated_prompt'
                out_annotated_prompt.mkdir(parents=True, exist_ok=True)

                model = init_model()

                for index, (result, frame_path) in enumerate(results_list):
                    if result.masks is None and results_list[index - 1][0].masks:
                        frame = Image.open(frame_path).convert("RGB").resize((224, 224))
                        detections = frame_annotations[frame_path.stem]

                        bboxes = detections.xyxy
                        # masks = detections.mask
                        # masks = binary_fill_holes(masks).astype(np.uint8)
                        # masks[masks > 0] = 1

                        if results_list[index - 1][0].masks:
                            masks = results_list[index - 1][0].masks.data.cpu().numpy().astype(np.uint8)

                        annotated_prompts = box_annotator.annotate(np.array(frame.copy()), detections)
                        cv2.imwrite(str(out_annotated_prompt / frame_path.name),
                                    cv2.cvtColor(annotated_prompts, cv2.COLOR_RGB2BGR))

                        prompts = {
                            "masks": masks,
                            "cls": np.array([0] * len(bboxes))  # Single class "polyp" as class 0
                        }

                        # Pass the visual prompt (bounding boxes) with the predictor
                        results = model.track(
                            source=np.array(frame),
                            imgsz=image_size,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            prompts=prompts,
                            predictor=YOLOEVPSegPredictor,
                            persist=True
                        )

                        if results[0].masks:
                            new_masks_generated = True

                        results_list[index] = (results[0], frame_path)

                        cv2.imwrite(str(out_annotated_mask / frame_path.name), results[0].plot())

            results_list.reverse()

            #out_annotated = out_dir_dataset / f'final_result'
            #out_annotated.mkdir(parents=True, exist_ok=True)

            # Fall back to attention map
            for index, (result, frame_path) in enumerate(results_list):
                if result.masks is None:
                    frame = Image.open(frame_path).convert("RGB").resize((224, 224))

                    detections = frame_annotations[frame_path.stem]
                    confidences = detections.confidence

                    bboxes = detections.xyxy
                    masks = detections.mask
                    masks = binary_fill_holes(masks).astype(np.uint8)
                    masks[masks > 0] = 1

                    bboxes = torch.tensor([[x1, y1, x2, y2, conf, 0] for (x1, y1, x2, y2), conf in zip(bboxes, confidences)])
                    masks = torch.tensor(masks)

                    result.update(boxes=bboxes, masks=masks)


            for index, (result, frame_path) in enumerate(results_list):
                plot_dir = out_dir_dataset / 'plot' / frame_path.parent.name
                plot_dir.mkdir(parents=True, exist_ok=True)
                mask_out_dir = out_dir_dataset / 'mask' / frame_path.parent.name
                mask_out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(plot_dir / frame_path.name), result.plot())
                combined_mask = np.zeros_like(result.orig_img)
                if result.masks:
                    masks = result.masks.data.cpu().numpy()
                    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

                cv2.imwrite(str((mask_out_dir / frame_path.name).with_suffix('.png')), combined_mask)


        #tracker = sv.ByteTrack(frame_rate=20)

        #for index, (result, frame_path) in enumerate(results_list):
        #    detections = sv.Detections.from_ultralytics(result)
        #    detections = tracker.update_with_detections(detections)

        #    image = Image.open(frame_path).convert("RGB").resize((224, 224))
        #    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
        #    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image.size)
        #    labels = [f"polyp {conf:.2f}" for conf in detections.confidence]
        #    annotated_image = image.copy()
        #    annotated_image = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4) \
        #        .annotate(scene=annotated_image, detections=detections)
        #    annotated_image = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness) \
        #        .annotate(scene=annotated_image, detections=detections)
        #    annotated_image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_scale=text_scale,
        #                                        smart_position=True) \
        #        .annotate(scene=annotated_image, detections=detections, labels=labels)
        #    cv2.imwrite(str(out_annotated / frame_path.name), cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR))
