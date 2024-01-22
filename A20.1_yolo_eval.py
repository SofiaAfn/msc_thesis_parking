# %%
import ultralytics
from ultralytics import YOLO
import numpy as np
# from PIL import Image
# import requests
# from io import BytesIO
import cv2
ultralytics.checks()
import supervision as sv
import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline
import json
import os

# %%
model = YOLO(
    '/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/notebooks/yolov8n.pt', 
    task='detect') 

# %%
def predictions(img_path):
    """
    Predict objects in the image using the YOLO model.

    Parameters:
    - img_path: Path to the input image.

    Returns:
    - detections: Detected objects in the image.
    """
    # predictions = model.predict(img, show=True, imgsz=(1253,705))
    # if format == 'RGB':
    #     # Skip the BGR to RGB conversion
    #     pass
    # else:
    #     # Convert from BGR to RGB
    #     img = img[..., ::-1]
    
    results = model.predict(img_path)
    # detections = sv.Detections.from_yolov8(predictions[0])
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int),
    )

    return results, detections

def box_annotate(img, detections):
    """
    Annotate the image with bounding boxes and labels.

    Parameters:
    - img_path: Path to the input image.
    - detections: Detected objects in the image.

    Returns:
    - annotated_image: Image with annotations.
    """

    # dict maping class_id to class_name
    CLASS_NAMES_DICT = model.model.names
    # class_ids of interest - car, motorcycle, bus and truck
    # CLASS_ID = [2, 3, 5, 7]
    CLASS_ID = [2]
    
    if isinstance(img, str):
        # input_data is a path to the image file
        img = cv2.imread(img)
        if img is None:
            raise ValueError(f"Failed to load image from {img}")
    elif isinstance(img, np.ndarray):
        # input_data is a numpy array
        img = img
        
   
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5, text_padding= 1)
    labels = [
        f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id in detections
    ]

    annotated_image = box_annotator.annotate(img, detections=detections, labels=labels)
    
    plt.figure(figsize=(100,100))
    plt.imshow(img, cmap="gray")
    plt.title("annotated")
    plt.axis("off")
    plt.show()

    return annotated_image

def process_and_display_image(img_path, x, y, w, h, display_original=True, display_cropped=True):
    """
    Load an image, crop it, and display the original and cropped images.

    Parameters:
    - img_path: Path to the input image.
    - x, y: Coordinates for the top-left corner of the cropping rectangle.
    - w, h: Width and height of the cropping rectangle.
    - display_original: Whether to display the original image.
    - display_cropped: Whether to display the cropped image.
    """
    # Load the image
   
    if isinstance(img_path, str):
        # input_data is a path to the image file
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image from {img_path}")
    elif isinstance(img_path, np.ndarray):
        # input_data is a numpy array
        img = img_path
        
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop the image
    crop_img = img[y:y+h, x:x+w]
    crop_img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

     #display_original:
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')

    #display_cropped:
    plt.subplot(1, 2, 2)
    plt.imshow(crop_img_rgb)
    plt.title('Cropped Image')

    plt.show()


    return crop_img_rgb

def save_detections_to_file(detections, save_path):
    with open(save_path, 'w') as file:
        json.dump(detections, file)

# %%

# Path to the folder containing images
image_folder = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_selected/"

image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Path to the folder where evaluated images will be saved
save_folder = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/"

crop_save_folder = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/crop/"

detections_save_folder = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/detections/"

gt_json_file = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/crop/kris_ground_truth_coco.json"


# %%
def write_detections_to_file(detections, image_width, image_height, output_file_path):
    """
    Write YOLOv8 detection results to a file in YOLO format.

    Parameters:
    detections (object): The detection result from YOLOv8.
    image_width (int): Width of the image on which detection was performed.
    image_height (int): Height of the image on which detection was performed.
    output_file_path (str): Path to the output file where results will be written.
    """
    with open(output_file_path, 'w') as file:
        for bbox, class_id in zip(detections.xyxy, detections.class_id):
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = bbox

            # Convert to YOLO format: [x_center, y_center, width, height], normalized
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # Write the formatted string to the file
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            


# %%

# Ensure the detections save folder exists
if not os.path.exists(detections_save_folder):
    os.makedirs(detections_save_folder)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    
    crop_img_rgb = process_and_display_image(image_path, x=0, y=400, w=1200, h=500)
    
    height, width = crop_img_rgb.shape[:2]
    print(f"H, W: {height},{width}")
    
    # Generate unique save path for each cropped image
    crop_save_path = os.path.join(crop_save_folder, image_file)
    
    # Save the cropped image
    cv2.imwrite(crop_save_path, crop_img_rgb)
    
    # res_org, detections_org = predictions(image_path)  # Detections on uncropped images
    res_crop,detections_crop = predictions(crop_img_rgb)  # Detections on cropped images
    
    # annotated_org = box_annotate(image_path, detections_org)
    annotated_crop = box_annotate(crop_img_rgb, detections_crop)
    
    # Define the output file path
    output_file_path = os.path.join(detections_save_folder, os.path.splitext(image_file)[0] + '_detections.txt')

    # Write detections to the file
    write_detections_to_file(detections_crop, width, height, output_file_path)
    print(f"Detections for {image_file} written to {output_file_path}")
    
    # # Save cropped image detections
    # detections_crop_save_path = os.path.join(detections_save_folder, f"crop_{image_file}.json")
    # save_detections_to_file(detections_crop, detections_crop_save_path)
    
    # # Generate unique save path for each annotated image
    # save_to_filename = os.path.join(save_folder, image_file)

    # # Save the annotated image
    # cv2.imwrite(save_to_filename, annotated_crop)

    # print(f"{image_file} Original: {res_org},{detections_org}")
    # print(f"{image_file}Croppd: {res_crop}, {detections_crop}")
    # print(f"{image_file} file saved")


# %%

def convert_coco_to_yolo(coco_json_path, output_dir, image_width, image_height):
    
    # Remove leading/trailing whitespaces in the file path
    coco_json_path = coco_json_path.strip()

    # Check if the file exists
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(f"The file {coco_json_path} does not exist.")
    
    with open(coco_json_path) as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img in data['images']:
        image_id = img['id']
        file_name = os.path.splitext(img['file_name'])[0] + '_gt.txt'
        annotations = [a for a in data['annotations'] if a['image_id'] == image_id]

        with open(os.path.join(output_dir, file_name), 'w') as file:
            for ann in annotations:
                # COCO format: [x_min, y_min, width, height]
                x_min, y_min, width, height = ann['bbox']

                # Convert to YOLO format: [x_center, y_center, width, height], normalized
                x_center = x_min + width / 2
                y_center = y_min + height / 2
                x_center /= image_width
                y_center /= image_height
                width /= image_width
                height /= image_height

                # Write to file
                file.write(f"{ann['category_id']} {x_center} {y_center} {width} {height}\n")

# Example usage
coco_json_path ="/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/crop/kris_ground_truth_coco.json "

gt_save_folder  = "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/crop/"

# width = 308  # Replace with your image width
# height = 1200  # Replace with your image height

convert_coco_to_yolo(coco_json_path, gt_save_folder, width, height)


# %%
def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1 (tuple): bounding box in format (x1, y1, x2, y2).
    box2 (tuple): bounding box in format (x1, y1, x2, y2).
    
    Returns:
    float: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


# %%
def read_yolo_format_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        _, x_center, y_center, width, height = map(float, line.split())
        boxes.append((x_center, y_center, width, height))  # Exclude class_id
    return boxes

def get_file_paths(folder, suffix):
    file_paths = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(suffix):
            key = file_name.replace(suffix, '')
            file_paths[key] = os.path.join(folder, file_name)
    return file_paths

def read_yolo_format_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def get_file_paths(folder, suffix):
    file_paths = {}
    for file_name in os.listdir(folder):
        if file_name.endswith(suffix):
            key = file_name.replace(suffix, '')
            file_paths[key] = os.path.join(folder, file_name)
    return file_paths


# %%
def calculate_iou_for_each_image(ground_truth_folder, detections_folder, image_folder):
    ground_truth_files = get_file_paths(ground_truth_folder, '_gt.txt')
    detection_files = get_file_paths(detections_folder, '_detections.txt')

    for image_key in ground_truth_files:
        if image_key in detection_files:
            ground_truth_boxes = read_yolo_format_file(ground_truth_files[image_key])
            detection_boxes = read_yolo_format_file(detection_files[image_key])

            # Assuming single object per image for simplicity
            if ground_truth_boxes and detection_boxes:
                gt_box = ground_truth_boxes[0][1:]  # Exclude class_id
                det_box = detection_boxes[0][1:]  # Exclude class_id
                iou = calculate_iou(gt_box, det_box)

                # Visualize
                image_path = os.path.join(image_folder, image_key + '.jpg')
                visualize_boxes_and_iou(image_path, gt_box, det_box, iou)


# %%
import cv2
import matplotlib.pyplot as plt

def visualize_boxes_and_iou(image_path, gt_boxes, det_boxes, ious):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    for gt_box, det_box, iou in zip(gt_boxes, det_boxes, ious):
        # Convert normalized coordinates to pixel coordinates
        gt_box_pixel = convert_to_pixel_coordinates(gt_box, w, h)
        det_box_pixel = convert_to_pixel_coordinates(det_box, w, h)

        # Draw boxes
        image = cv2.rectangle(image, (gt_box_pixel[0], gt_box_pixel[1]), (gt_box_pixel[2], gt_box_pixel[3]), (0, 255, 0), 2)
        image = cv2.rectangle(image, (det_box_pixel[0], det_box_pixel[1]), (det_box_pixel[2], det_box_pixel[3]), (255, 0, 0), 2)

        # Display IoU
        cv2.putText(image, f'IoU: {iou:.2f}', (det_box_pixel[0], det_box_pixel[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    plt.imshow(image)
    plt.show()


def convert_to_pixel_coordinates(box, width, height):
    x_center, y_center, w, h = box
    x_min = int((x_center - w / 2) * width)
    y_min = int((y_center - h / 2) * height)
    x_max = int((x_center + w / 2) * width)
    y_max = int((y_center + h / 2) * height)
    return x_min, y_min, x_max, y_max
 


# %%
#
# calculate_iou_for_each_image(gt_save_folder, detections_save_folder, image_folder)



# %%
def process_image(image_path, gt_file, det_file):
    gt_data = read_yolo_format_file(gt_file)
    det_data = read_yolo_format_file(det_file)

    ious = []
    for gt in gt_data:
        gt_box = gt[1:]  # Extract only the bounding box coordinates
        for det in det_data:
            det_box = det[1:]  # Extract only the bounding box coordinates
            iou = calculate_iou(gt_box, det_box)
            ious.append(iou)

    visualize_boxes_and_iou(image_path, [box[1:] for box in gt_data], [box[1:] for box in det_data], ious)



# %%
def process_matching_files(image_folder, ground_truth_folder, detections_folder):
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base_filename = os.path.splitext(filename)[0]
        gt_file = os.path.join(ground_truth_folder, base_filename + '_gt.txt')
        det_file = os.path.join(detections_folder, base_filename + '_detections.txt')

        if os.path.exists(gt_file) and os.path.exists(det_file):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path, gt_file, det_file)


# %%
process_matching_files(crop_save_folder, gt_save_folder, detections_save_folder)


# %%
import matplotlib.pyplot as plt

def visualize_detections(image_path, gt_boxes, det_boxes, iou_threshold=0.5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    legend = {
        'Green': 'Ground Truth',
        'Red': 'False Positive',
        'Blue': 'True Positive',
        'Orange': 'False Positive',
        #'Yellow': 'False Negative'
    }

    tp, tn, fp, fn = 0, 0, 0, 0

    for gt_box in gt_boxes:
        class_id, *gt_box_coords = gt_box
        gt_box_pixel = convert_to_pixel_coordinates(gt_box_coords, w, h)
        # Draw bounding box in green for ground truth
        image = cv2.rectangle(image, (gt_box_pixel[0], gt_box_pixel[1]), (gt_box_pixel[2], gt_box_pixel[3]), (0, 128, 0), 2)

    for det_box in det_boxes:
        class_id, *det_box_coords = det_box
        det_box_pixel = convert_to_pixel_coordinates(det_box_coords, w, h)
        matched = False
        iou_max = 0.0

        for gt_box in gt_boxes:
            gt_class_id, *gt_box_coords = gt_box
            if class_id != gt_class_id:
                continue  # Skip if class IDs do not match
            iou = calculate_iou(gt_box_coords, det_box_coords)

            if iou >= iou_threshold and iou > iou_max:
                matched = True
                iou_max = iou

        if matched:
            # True Positive: Blue box
            image = cv2.rectangle(image, (det_box_pixel[0], det_box_pixel[1]), (det_box_pixel[2], det_box_pixel[3]), (0, 0, 255), 2)
            tp += 1
        else:
            # False Positive: Red box
            image = cv2.rectangle(image, (det_box_pixel[0], det_box_pixel[1]), (det_box_pixel[2], det_box_pixel[3]), (255, 0, 0), 2)
            fp += 1

    # False Negatives are ground truth boxes without a matching detection
    for gt_box in gt_boxes:
        class_id, *gt_box_coords = gt_box
        gt_box_pixel = convert_to_pixel_coordinates(gt_box_coords, w, h)
        if not any(calculate_iou(gt_box_coords, det_box_coords) >= iou_threshold for det_box_coords in [box[1:] for box in det_boxes if box[0] == class_id]):
            # Draw a yellow box for false negatives
            # image = cv2.rectangle(image, (gt_box_pixel[0], gt_box_pixel[1]), (gt_box_pixel[2], gt_box_pixel[3]), (255, 255, 0), 2)
            fn += 1

    # Calculate True Negatives (TN)
    tn = len(det_boxes) - tp - fp

    # Create a figure with extra space on the right for the legend
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    
    # Draw the legend outside the image in the white space
    legend_handles = [plt.Line2D([0], [0], color=color, label=f'{description}') for color, description in legend.items()]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1, 1))
    
    # # Display TP, TN, FP, FN counts below the x-axis
    # ax.text(0.1, -0.1, f'TP: {tp}', transform=ax.transAxes, fontsize=12, color='black')
    # ax.text(0.3, -0.1, f'TN: {tn}', transform=ax.transAxes, fontsize=12, color='black')
    # ax.text(0.5, -0.1, f'FP: {fp}', transform=ax.transAxes, fontsize=12, color='black')
    # ax.text(0.7, -0.1, f'FN: {fn}', transform=ax.transAxes, fontsize=12, color='black')
    # Display TP, TN, FP, FN counts in the bottom right corner
    tp_text = f'TP: {tp}'
    tn_text = f'TN: {tn}'
    fp_text = f'FP: {fp}'
    fn_text = f'FN: {fn}'
    text = f'{tp_text}\n{tn_text}\n{fp_text}\n{fn_text}'
    ax.text(1.05, 0.005, text, fontsize=10, transform=ax.transAxes, verticalalignment='bottom')

  

    plt.show()


# %%
def read_yolo_format_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = line.split()
        class_id = int(parts[0])  # Extract class_id
        box_data = list(map(float, parts[1:]))  # Extract remaining values as (x_center, y_center, width, height)
        boxes.append((class_id, *box_data))
    return boxes


# %%
import os
import matplotlib.pyplot as plt

# Define a function to process and visualize images from different folders
def process_and_visualize_images(image_folder, gt_folder, detections_folder, output_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        gt_file = os.path.join(gt_folder, f"{os.path.splitext(image_file)[0]}_gt.txt")
        det_file = os.path.join(detections_folder, f"{os.path.splitext(image_file)[0]}_detections.txt")

        if os.path.exists(gt_file) and os.path.exists(det_file):
            print(f"Processing image: {image_file}")
            visualize_detections(image_path, read_yolo_format_file(gt_file), read_yolo_format_file(det_file))

            # Define the output image file path
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_output.jpg")
            print(f"Saving output image: {output_image_path}")

            # Save the output image
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)

            # Close the current plot
            plt.close()




# %%
evals_save_folder =  "/home/sf_afn/Insync/sofiaa720@gmail.com/Google Drive/masters_thesis/code_v1/msc_parking/univrses/data/eval_subset_samples/kris_evaluated/evals"

process_and_visualize_images(crop_save_folder, gt_save_folder, detections_save_folder, evals_save_folder)


# %%



