import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, font as tkFont

# --- Core Logic ---

try:
    detector = MTCNN()
except Exception as e:
    print(f"Error initializing MTCNN: {e}")
    print("Please ensure MTCNN and its backend (e.g., TensorFlow) are correctly installed.")
    detector = None

def is_skin_tone(face_region):
    if face_region is None or face_region.size == 0: return False
    try:
        ycrcb = cv.cvtColor(face_region, cv.COLOR_BGR2YCrCb)
    except cv.error as e:
        print(f"Error converting to YCrCb: {e}"); return False
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return np.mean(skin_mask) > 0.3

def detect_and_crop_faces(image_path, padding_px, combine_output, make_bw,
                          confidence_thresh, bypass_keypoints, bypass_skin_tone,
                          status_callback):
    if detector is None:
        status_callback("Error: MTCNN detector not initialized. Cannot process.\n")
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

        if confidence < confidence_thresh:
            status_callback(f"    -> Skipped: Low confidence ({confidence:.3f} < {confidence_thresh:.2f}).\n")
            continue

        if not bypass_keypoints:
            required_keys = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
            if not all(k in keypoints and keypoints[k] is not None for k in required_keys):
                missing_kps = [k for k in required_keys if k not in keypoints or keypoints[k] is None]
                status_callback(f"    -> Skipped: Missing keypoints: {', '.join(missing_kps)}.\n")
                continue
        
        x1, y1, x2, y2 = max(0, x - padding_px), max(0, y - padding_px), min(img_w, x + w + padding_px), min(img_h, y + h + padding_px)
        if x2 <= x1 or y2 <= y1:
            status_callback(f"    -> Skipped: Invalid crop dimensions after padding.\n")
            continue
        face_crop_bgr = image[y1:y2, x1:x2]

        passed_skin_check = bypass_skin_tone or is_skin_tone(face_crop_bgr)
        if not passed_skin_check:
            status_callback(f"    -> Skipped: Failed skin-tone validation.\n")
            continue 

        current_face_to_store = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2GRAY) if make_bw else face_crop_bgr
        valid_face_crops.append(current_face_to_store)
        status_callback(f"    -> Validated and added to list.\n")

    if not valid_face_crops:
        status_callback("No valid faces passed the verification process.\n")
        return

    base, ext = os.path.splitext(image_path)
    if combine_output:
        if not valid_face_crops: status_callback("No faces to combine.\n"); return
        common_height = 150
        processed_for_concat = []
        for crop in valid_face_crops:
            h_crop, w_crop = crop.shape[:2]; scale = common_height / h_crop
            resized_crop = cv.resize(crop, (int(w_crop * scale), common_height))
            processed_for_concat.append(resized_crop)
        try:
            combined_image = cv.hconcat(processed_for_concat)
            output_path = f"{base}_combined{ext}"
            cv.imwrite(output_path, combined_image)
            status_callback(f"Saved combined faces to: {output_path}\n")
        except cv.error as e:
            status_callback(f"Error combining images: {e}\nSaving individually.\n")
            for i, crop_to_save in enumerate(valid_face_crops):
                output_path = f"{base}_face_{i+1}{ext}"
                cv.imwrite(output_path, crop_to_save)
                status_callback(f"Saved: {output_path}\n")
    else:
        for i, crop_to_save in enumerate(valid_face_crops):
            output_path = f"{base}_face_{i+1}{ext}"
            cv.imwrite(output_path, crop_to_save)
            status_callback(f"Saved: {output_path}\n")
    status_callback("Processing finished.\n")


# --- Tkinter UI ---
class FaceCropperApp:
    def __init__(self, root):
        self.root = root
        root.title("MTCNN Face Detector & Cropper")
        root.geometry("750x700")

        # Configure styles for ttk widgets
        style = ttk.Style()
        style.theme_use('classic') # 'clam', 'alt', 'default', 'classic' 
        style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        style.configure("TLabel", padding=3, font=('Helvetica', 10))
        style.configure("TCheckbutton", padding=3, font=('Helvetica', 10))
        style.configure("TEntry", padding=5, font=('Helvetica', 10))
        style.configure("TLabelframe.Label", font=('Helvetica', 11, 'bold'))
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'), padding=(0,10,0,5))


        # --- Main PanedWindow for layout flexibility ---
        main_paned_window = ttk.PanedWindow(root, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Top Frame for Title and Description ---
        header_frame = ttk.Frame(main_paned_window, padding=10)
        main_paned_window.add(header_frame, weight=0) # Don't let this pane resize much

        app_title_label = ttk.Label(header_frame, text="Face Detector & Cropper", style="Header.TLabel")
        app_title_label.pack(pady=(0, 5))

        # Program Purpose Description
        purpose_text_area = tk.Text(header_frame, height=4, wrap=tk.WORD, relief=tk.FLAT,
                                     bg=root.cget('bg'), # Match root background
                                     font=('Helvetica', 10), padx=5)
        purpose_text_area.insert(tk.END, 
            "This application uses MTCNN to detect faces in images. "
            "You can adjust detection parameters, apply padding, "
            "and choose output formats like combined images or black & white crops. "
            "Ideal for preparing face datasets or quick facial analysis. "
            "An application created for Digital Image Processing - Final Project"
        )
        purpose_text_area.configure(state='disabled') # Make it read-only
        purpose_text_area.pack(fill=tk.X, pady=(0, 10))


        # --- Middle Frame for Controls ---
        controls_paned_window = ttk.PanedWindow(main_paned_window, orient=tk.HORIZONTAL)
        main_paned_window.add(controls_paned_window, weight=1) # Allow resizing

        # Left side of middle frame: File and Basic Options
        left_controls_frame = ttk.Frame(controls_paned_window, padding=10)
        controls_paned_window.add(left_controls_frame, weight=1)

        # File Selection Section
        file_frame = ttk.LabelFrame(left_controls_frame, text="Image Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        self.file_label_var = tk.StringVar(value="No image selected")
        self.file_label = ttk.Label(file_frame, textvariable=self.file_label_var, wraplength=300)
        self.file_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10))
        
        self.select_button = ttk.Button(file_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.RIGHT)

        # Basic Options Section
        basic_options_frame = ttk.LabelFrame(left_controls_frame, text="Output Options", padding=10)
        basic_options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Padding
        padding_frame = ttk.Frame(basic_options_frame)
        padding_frame.pack(fill=tk.X, pady=3)
        ttk.Label(padding_frame, text="Padding (px):").pack(side=tk.LEFT)
        self.padding_var = tk.IntVar(value=10)
        self.padding_entry = ttk.Entry(padding_frame, textvariable=self.padding_var, width=7, justify=tk.RIGHT)
        self.padding_entry.pack(side=tk.LEFT, padx=(5,0))

        # Checkboxes
        self.combine_var = tk.BooleanVar(value=False)
        self.combine_check = ttk.Checkbutton(basic_options_frame, text="Combine Output into One Image", variable=self.combine_var)
        self.combine_check.pack(anchor=tk.W, pady=3)

        self.bw_var = tk.BooleanVar(value=False)
        self.bw_check = ttk.Checkbutton(basic_options_frame, text="Convert Cropped Faces to B&W", variable=self.bw_var)
        self.bw_check.pack(anchor=tk.W, pady=3)

        # Right side of middle frame: Advanced Options
        right_controls_frame = ttk.Frame(controls_paned_window, padding=10)
        controls_paned_window.add(right_controls_frame, weight=1)

        # Advanced/Debug Options Section
        advanced_options_frame = ttk.LabelFrame(right_controls_frame, text="Advanced Detection Settings", padding=10)
        advanced_options_frame.pack(fill=tk.BOTH, expand=True)
        
        # Confidence
        confidence_frame = ttk.Frame(advanced_options_frame)
        confidence_frame.pack(fill=tk.X, pady=3)
        ttk.Label(confidence_frame, text="Confidence (0-1):").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.90)
        self.confidence_entry = ttk.Entry(confidence_frame, textvariable=self.confidence_var, width=7, justify=tk.RIGHT)
        self.confidence_entry.pack(side=tk.LEFT, padx=(5,0))

        # Bypass Checkboxes
        self.bypass_keypoints_var = tk.BooleanVar(value=False)
        self.bypass_keypoints_check = ttk.Checkbutton(advanced_options_frame, text="Bypass Keypoint Validation", variable=self.bypass_keypoints_var)
        self.bypass_keypoints_check.pack(anchor=tk.W, pady=3)

        self.bypass_skin_tone_var = tk.BooleanVar(value=False)
        self.bypass_skin_tone_check = ttk.Checkbutton(advanced_options_frame, text="Bypass Skin Tone Validation", variable=self.bypass_skin_tone_var)
        self.bypass_skin_tone_check.pack(anchor=tk.W, pady=3)
        
        # Process Button (centered below controls)
        process_button_frame = ttk.Frame(left_controls_frame) # Put it in the left pane for now
        process_button_frame.pack(pady=20)
        self.process_button = ttk.Button(process_button_frame, text="✨ Process Image ✨", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack()


        # --- Bottom Frame for Log ---
        log_frame = ttk.LabelFrame(main_paned_window, text="Processing Log", padding=10)
        main_paned_window.add(log_frame, weight=2) # Give log more space

        self.status_text = scrolledtext.ScrolledText(log_frame, height=10, relief=tk.SUNKEN, borderwidth=1,
                                                     wrap=tk.WORD, font=('Consolas', 9) ) # Monospaced font for log
        self.status_text.pack(fill=tk.BOTH, expand=True)
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
            self.file_label_var.set(os.path.basename(path)) # Update stringvar
            self.process_button.config(state=tk.NORMAL)
            self._update_status(f"Selected: {path}\n")
        else:
            self.file_label_var.set("No image selected")
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
            if padding < 0: messagebox.showerror("Error", "Padding cannot be negative."); return
            
            confidence_thresh = self.confidence_var.get()
            if not (0.0 <= confidence_thresh <= 1.0):
                messagebox.showerror("Error", "Confidence threshold must be between 0.0 and 1.0."); return

        except tk.TclError:
            messagebox.showerror("Error", "Invalid padding or confidence value. Must be a number."); return

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
        root_temp = tk.Tk()
        root_temp.withdraw() # Hide the dummy root window
        messagebox.showerror("Startup Error", "MTCNN failed to initialize. The application cannot run.\nPlease check console for TensorFlow/PyTorch errors and ensure MTCNN is installed correctly.")
        # sys.exit(1) # Or just let it close
    else:
        root = tk.Tk()
        app = FaceCropperApp(root)
        root.mainloop()