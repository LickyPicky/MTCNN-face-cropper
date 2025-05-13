import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk # ttk for separator

# --- Core Logic (modified) ---

try:
    detector = MTCNN()
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    print("Please ensure MTCNN and its backend (e.g., TensorFlow) are correctly installed.")
    detector = None

def is_skin_tone(face_region):
    """Verify if a region contains skin-tone pixels using YCrCb color filtering."""
    if face_region is None or face_region.size == 0:
        return False
    try:
        ycrcb = cv.cvtColor(face_region, cv.COLOR_BGR2YCrCb)
    except cv.error as e:
        print(f"Error converting to YCrCb (possibly empty region): {e}")
        return False
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return np.mean(skin_mask) > 0.3

def detect_and_crop_faces(image_path, padding_px, combine_output, make_bw,
                          confidence_thresh, bypass_keypoints, bypass_skin_tone,
                          status_callback):
    """
    Detect faces, apply padding, validate (with bypass options), and save/combine.
    """
    if detector is None:
        status_callback("Error: MTCNN detector not initialized. Cannot process.\n")
        # UI should also show an error if this happens at processing time
        return

    status_callback(f"Processing {image_path}...\n")
    
    image = cv.imread(image_path)
    if image is None:
        status_callback(f"Error: Could not load image: {image_path}\n")
        messagebox.showerror("Image Error", f"Could not load image: {image_path}")
        return
    
    rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_image)

    if not detections:
        status_callback("No faces detected by MTCNN.\n")
        return

    valid_face_crops = []
    img_h, img_w = image.shape[:2]

    status_callback(f"MTCNN found {len(detections)} raw candidates.\n")

    for i, detection in enumerate(detections):
        x, y, w, h = detection['box']
        confidence = detection['confidence']
        keypoints = detection['keypoints']

        status_callback(f"  Candidate {i+1}: Confidence={confidence:.3f}, Box=[{x},{y},{w},{h}]\n")

        # 1. Confidence threshold
        if confidence < confidence_thresh:
            status_callback(f"    -> Skipped: Low confidence ({confidence:.3f} < {confidence_thresh:.2f}).\n")
            continue

        # 2. Validate keypoints (if not bypassed)
        if not bypass_keypoints:
            required_keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
            # Check if all required keypoints are present in the 'keypoints' dict from MTCNN
            # MTCNN's keypoints dict itself might not contain a key if it couldn't detect it.
            # And then, their positions should also be valid (though MTCNN usually gives positions if key is present)
            if not all(k in keypoints and keypoints[k] is not None for k in required_keys):
                missing_kps = [k for k in required_keys if k not in keypoints or keypoints[k] is None]
                status_callback(f"    -> Skipped: Missing keypoints: {', '.join(missing_kps)}.\n")
                continue
        # No specific "else" log here as the bypass is logged once at the start of processing

        # Apply padding
        x1 = max(0, x - padding_px)
        y1 = max(0, y - padding_px)
        x2 = min(img_w, x + w + padding_px)
        y2 = min(img_h, y + h + padding_px)

        if x2 <= x1 or y2 <= y1: # Should not happen if x,y,w,h are valid
            status_callback(f"    -> Skipped: Invalid crop dimensions after padding.\n")
            continue
            
        face_crop_bgr = image[y1:y2, x1:x2] # This is the BGR crop after padding

        # 3. Validate skin-tone (if not bypassed)
        passed_skin_check = False
        if bypass_skin_tone:
            passed_skin_check = True
        elif is_skin_tone(face_crop_bgr): # Use the BGR crop for skin tone
            passed_skin_check = True
        
        if not passed_skin_check: # Only if not bypassed AND skin tone check failed
            status_callback(f"    -> Skipped: Failed skin-tone validation.\n")
            continue 

        # If all checks passed (or were bypassed successfully)
        current_face_to_store = face_crop_bgr # Start with the BGR crop
        if make_bw:
            current_face_to_store = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2GRAY)
        
        valid_face_crops.append(current_face_to_store)
        status_callback(f"    -> Validated and added to list.\n")


    if not valid_face_crops:
        status_callback("No valid faces passed the verification process.\n")
        return

    base, ext = os.path.splitext(image_path)

    if combine_output:
        if not valid_face_crops: # Should be redundant due to check above, but good practice
            status_callback("No faces to combine.\n")
            return

        common_height = 150
        processed_for_concat = []
        for crop in valid_face_crops:
            h_crop, w_crop = crop.shape[:2]
            scale = common_height / h_crop
            new_w = int(w_crop * scale)
            resized_crop = cv.resize(crop, (new_w, common_height))
            processed_for_concat.append(resized_crop)

        try:
            combined_image = cv.hconcat(processed_for_concat)
            output_path = f"{base}_combined{ext}"
            cv.imwrite(output_path, combined_image)
            status_callback(f"Saved combined faces to: {output_path}\n")
        except cv.error as e:
            status_callback(f"Error combining images: {e}\n")
            status_callback("Saving faces individually instead.\n")
            for i, crop_to_save in enumerate(valid_face_crops):
                output_path = f"{base}_face_{i+1}{ext}"
                cv.imwrite(output_path, crop_to_save)
                status_callback(f"Saved: {output_path}\n")
    else: # Save individually
        for i, crop_to_save in enumerate(valid_face_crops):
            output_path = f"{base}_face_{i+1}{ext}"
            cv.imwrite(output_path, crop_to_save)
            status_callback(f"Saved: {output_path}\n")
    
    status_callback("Processing finished.\n")

# --- Tkinter UI ---
class FaceCropperApp:
    def __init__(self, root):
        self.root = root
        root.title("Face Detector & Cropper")

        # Frame for file selection
        file_frame = tk.Frame(root, padx=10, pady=5)
        file_frame.pack(fill=tk.X)

        self.file_label = tk.Label(file_frame, text="No image selected")
        self.file_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.select_button = tk.Button(file_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.RIGHT)

        # Frame for basic options
        basic_options_frame = tk.Frame(root, padx=10, pady=5)
        basic_options_frame.pack(fill=tk.X)

        tk.Label(basic_options_frame, text="Padding (px):").pack(side=tk.LEFT)
        self.padding_var = tk.IntVar(value=10)
        self.padding_entry = tk.Entry(basic_options_frame, textvariable=self.padding_var, width=5)
        self.padding_entry.pack(side=tk.LEFT, padx=(0, 10))

        self.combine_var = tk.BooleanVar(value=False)
        self.combine_check = tk.Checkbutton(basic_options_frame, text="Combine Output", variable=self.combine_var)
        self.combine_check.pack(side=tk.LEFT, padx=(0, 10))

        self.bw_var = tk.BooleanVar(value=False)
        self.bw_check = tk.Checkbutton(basic_options_frame, text="Make B&W", variable=self.bw_var)
        self.bw_check.pack(side=tk.LEFT)

        # Separator
        ttk.Separator(root, orient='horizontal').pack(fill='x', padx=10, pady=5)

        # Frame for Advanced/Debug options
        advanced_options_frame = tk.Frame(root, padx=10, pady=0) # Less pady at top
        advanced_options_frame.pack(fill=tk.X, pady=(0,5))
        
        tk.Label(advanced_options_frame, text="Advanced/Debug Options:").pack(anchor=tk.W, pady=(0,3))

        adv_row1_frame = tk.Frame(advanced_options_frame)
        adv_row1_frame.pack(fill=tk.X)
        tk.Label(adv_row1_frame, text="Confidence (0-1):").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.90) # Default confidence
        self.confidence_entry = tk.Entry(adv_row1_frame, textvariable=self.confidence_var, width=5)
        self.confidence_entry.pack(side=tk.LEFT, padx=(0, 20))

        adv_row2_frame = tk.Frame(advanced_options_frame) # New row for checkboxes
        adv_row2_frame.pack(fill=tk.X, pady=(3,0))
        self.bypass_keypoints_var = tk.BooleanVar(value=False)
        self.bypass_keypoints_check = tk.Checkbutton(adv_row2_frame, text="Bypass Keypoints Req.", variable=self.bypass_keypoints_var)
        self.bypass_keypoints_check.pack(side=tk.LEFT, padx=(0, 10))

        self.bypass_skin_tone_var = tk.BooleanVar(value=False)
        self.bypass_skin_tone_check = tk.Checkbutton(adv_row2_frame, text="Bypass Skin Tone Req.", variable=self.bypass_skin_tone_var)
        self.bypass_skin_tone_check.pack(side=tk.LEFT)
        
        # Process button
        self.process_button = tk.Button(root, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        # Status/Log area
        tk.Label(root, text="Log:").pack(anchor=tk.W, padx=10)
        self.status_text = scrolledtext.ScrolledText(root, height=10, width=70, wrap=tk.WORD)
        self.status_text.pack(padx=10, pady=(0,10), fill=tk.BOTH, expand=True)
        self.status_text.configure(state='disabled')

        self.image_path = None

    def _update_status(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
        self.status_text.configure(state='disabled')
        self.root.update_idletasks()

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        if path:
            self.image_path = path
            self.file_label.config(text=os.path.basename(path))
            self.process_button.config(state=tk.NORMAL)
            self._update_status(f"Selected: {path}\n")
        else:
            self.file_label.config(text="No image selected")
            self.process_button.config(state=tk.DISABLED)

    def process_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return
        
        if detector is None:
             messagebox.showerror("MTCNN Error", "MTCNN could not be initialized. Check console.")
             self._update_status("MTCNN not available. Aborting.\n")
             return

        try:
            padding = self.padding_var.get()
            if padding < 0:
                messagebox.showerror("Error", "Padding cannot be negative.")
                return
            
            confidence_thresh = self.confidence_var.get()
            if not (0.0 <= confidence_thresh <= 1.0):
                messagebox.showerror("Error", "Confidence threshold must be between 0.0 and 1.0.")
                return

        except tk.TclError: # Catches errors from .get() if value is not of correct type
            messagebox.showerror("Error", "Invalid padding or confidence value. Must be a number.")
            return

        combine = self.combine_var.get()
        make_bw = self.bw_var.get()
        bypass_keypoints = self.bypass_keypoints_var.get()
        bypass_skin_tone = self.bypass_skin_tone_var.get()

        self.process_button.config(state=tk.DISABLED)
        self._update_status("--- Starting processing ---\n")
        self._update_status(f"Settings: Padding={padding}px, Combine={combine}, B&W={make_bw}\n")
        self._update_status(f"Confidence Threshold: {confidence_thresh:.2f}\n")
        if bypass_keypoints: self._update_status("Keypoint validation will be BYPASSED.\n")
        if bypass_skin_tone: self._update_status("Skin tone validation will be BYPASSED.\n")
        
        try:
            detect_and_crop_faces(
                self.image_path, padding, combine, make_bw, 
                confidence_thresh, bypass_keypoints, bypass_skin_tone, 
                self._update_status
            )
        except Exception as e:
            self._update_status(f"An unexpected error occurred during processing: {e}\n")
            import traceback
            self._update_status(traceback.format_exc() + "\n")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            self.process_button.config(state=tk.NORMAL)
            self._update_status("--- Processing complete ---\n\n")


if __name__ == "__main__":
    if detector is None:
        # UI will show an error if user tries to process.
        pass 
        # For a critical failure at startup, you might show a messagebox here
        # before even creating the main window, or make the process button disabled
        # from the start with a message.

    root = tk.Tk()
    app = FaceCropperApp(root)
    root.mainloop()