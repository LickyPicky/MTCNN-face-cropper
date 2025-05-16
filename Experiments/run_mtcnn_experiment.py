import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import os
import pandas as pd # For reading CSV and organizing results

# Suppress TensorFlow logs (optional, place at the very top if used broadly)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Helper Function (from above) ---
def calculate_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6) # Add epsilon for stability
    return iou

# --- Main Experimental Logic ---
def run_experiment(image_dir, gt_file, confidence_thresholds, iou_threshold=0.5):
    """
    Runs the MTCNN detection experiment for different confidence thresholds.
    """
    try:
        detector = MTCNN()
    except Exception as e:
        print(f"Failed to initialize MTCNN: {e}")
        return None

    print(f"Loading ground truth from: {gt_file}")
    gt_df = pd.read_csv(gt_file)
    # Group ground truth by image filename
    gt_by_image = gt_df.groupby('image_filename')

    overall_results = []

    for conf_thresh in confidence_thresholds:
        print(f"\n--- Testing with Confidence Threshold: {conf_thresh:.2f} ---")
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt_faces = 0

        for image_filename in gt_df['image_filename'].unique():
            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping.")
                continue

            image = cv.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue
            
            rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            detections = detector.detect_faces(rgb_image)

            # Filter detections by current confidence threshold
            # MTCNN 'box' is [x, y, width, height]
            # We need [x_min, y_min, x_max, y_max] for IoU
            pred_boxes = []
            for det in detections:
                if det['confidence'] >= conf_thresh:
                    x, y, w, h = det['box']
                    pred_boxes.append([x, y, x + w, y + h])

            # Get ground truth boxes for this image
            try:
                current_gt_boxes_df = gt_by_image.get_group(image_filename)
                # Convert GT to list of [x_min, y_min, x_max, y_max], handling NaNs for no-face images
                gt_boxes_for_image = []
                if not current_gt_boxes_df[['x_min', 'y_min', 'x_max', 'y_max']].isnull().all().all():
                     gt_boxes_for_image = current_gt_boxes_df[['x_min', 'y_min', 'x_max', 'y_max']].values.tolist()
            except KeyError: # Image might be in directory but not in GT (e.g. a distractor) or vice-versa
                gt_boxes_for_image = []
            
            num_gt_in_image = len(gt_boxes_for_image)
            total_gt_faces += num_gt_in_image

            # Match predictions to ground truth
            tp_in_image = 0
            fp_in_image = 0
            
            # Keep track of which GT boxes have been matched
            gt_matched_flags = [False] * num_gt_in_image

            if pred_boxes: # If there are any predictions for this image
                for p_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for i, gt_box in enumerate(gt_boxes_for_image):
                        iou = calculate_iou(p_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched_flags[best_gt_idx]:
                        tp_in_image += 1
                        gt_matched_flags[best_gt_idx] = True # Mark this GT as matched
                    else:
                        fp_in_image += 1 # Either IoU too low or matched a GT already matched by a better pred (less likely here)
            
            fn_in_image = num_gt_in_image - sum(gt_matched_flags) # Unmatched GT boxes

            total_tp += tp_in_image
            total_fp += fp_in_image
            total_fn += fn_in_image
            # print(f"  Image: {image_filename}, GTs: {num_gt_in_image}, Preds: {len(pred_boxes)}, TPs: {tp_in_image}, FPs: {fp_in_image}, FNs: {fn_in_image}")


        # Calculate metrics for this confidence threshold
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0 # or total_gt_faces
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  Total GT Faces: {total_gt_faces}")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1_score:.4f}")
        
        overall_results.append({
            'confidence_threshold': conf_thresh,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Total_GT_Faces': total_gt_faces
        })
    
    return pd.DataFrame(overall_results)

if __name__ == "__main__":
    IMAGE_DIRECTORY = "test_images"  # Your folder with test images
    GROUND_TRUTH_FILE = "ground_truth.csv" # Your CSV annotation file
    
    # Confidence thresholds to test
    CONFIDENCE_LEVELS_TO_TEST = [0.50, 0.70, 0.80, 0.90, 0.95]
    IOU_MATCH_THRESHOLD = 0.5 # Standard IoU threshold for a match

    if not os.path.exists(IMAGE_DIRECTORY) or not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Error: Ensure '{IMAGE_DIRECTORY}' and '{GROUND_TRUTH_FILE}' exist.")
        print("Please create a 'test_images' folder with images and a 'ground_truth.csv' file.")
        print("Example ground_truth.csv format: image_filename,x_min,y_min,x_max,y_max")
        # Create dummy files for a quick test if they don't exist
        if not os.path.exists(IMAGE_DIRECTORY): os.makedirs(IMAGE_DIRECTORY)
        if not os.path.exists(GROUND_TRUTH_FILE):
            dummy_gt_data = {'image_filename': ['dummy.jpg', 'dummy.jpg'],
                             'x_min': [10, 100], 'y_min': [10, 100],
                             'x_max': [60, 150], 'y_max': [60, 150]}
            pd.DataFrame(dummy_gt_data).to_csv(GROUND_TRUTH_FILE, index=False)
            # You'd need to create a dummy.jpg in test_images too.
            print("Created dummy ground_truth.csv. Please replace with real data and images.")
    else:
        results_df = run_experiment(IMAGE_DIRECTORY, GROUND_TRUTH_FILE, 
                                    CONFIDENCE_LEVELS_TO_TEST, IOU_MATCH_THRESHOLD)

        if results_df is not None:
            print("\n--- Final Results Summary ---")
            print(results_df.to_string(index=False))
            
            # You can save this to a CSV
            results_df.to_csv("mtcnn_confidence_experiment_results.csv", index=False)
            print("\nResults saved to mtcnn_confidence_experiment_results.csv")

            # Optional: Basic Plotting (requires matplotlib)
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(results_df['confidence_threshold'], results_df['Precision'], marker='o', label='Precision')
                plt.plot(results_df['confidence_threshold'], results_df['Recall'], marker='s', label='Recall')
                plt.plot(results_df['confidence_threshold'], results_df['F1-Score'], marker='^', label='F1-Score')
                plt.xlabel("Confidence Threshold")
                plt.ylabel("Score")
                plt.title("MTCNN Performance vs. Confidence Threshold")
                plt.legend()
                plt.grid(True)
                plt.ylim(0, 1.05)
                plt.savefig("mtcnn_performance_plot.png")
                print("Performance plot saved to mtcnn_performance_plot.png")
                # plt.show() # Uncomment to display plot
            except ImportError:
                print("Matplotlib not installed. Skipping plot generation. (pip install matplotlib)")
