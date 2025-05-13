import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, font as tkFont, colorchooser

# --- Core Logic (remains the same as your previous version) ---
# ... (is_skin_tone function remains the same) ...
# ... (MTCNN initialization remains the same) ...

# (Make sure is_skin_tone and MTCNN initialization are here as in the previous version)
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
                          draw_on_original, shape_type, draw_color_bgr, # New params
                          status_callback):
    if detector is None:
        status_callback("Error: MTCNN detector not initialized. Cannot process.\n")
        return

    status_callback(f"Processing {image_path}...\n")
    original_image = cv.imread(image_path) # Load original
    if original_image is None:
        status_callback(f"Error: Could not load image: {image_path}\n")
        messagebox.showerror("Image Error", f"Could not load image: {image_path}")
        return
    
    # Work on a copy for drawing if needed, or for MTCNN processing
    image_to_process = original_image.copy()
    output_image_with_drawings = original_image.copy() # For drawing detections

    rgb_image = cv.cvtColor(image_to_process, cv.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_image)

    if not detections:
        status_callback("No faces detected by MTCNN.\n")
        return

    valid_face_crops = []
    validated_boxes_for_drawing = [] # Store (x,y,w,h) of validated faces for drawing
    img_h, img_w = image_to_process.shape[:2]
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
        
        # Bounding box for cropping (with padding)
        crop_x1 = max(0, x - padding_px)
        crop_y1 = max(0, y - padding_px)
        crop_x2 = min(img_w, x + w + padding_px)
        crop_y2 = min(img_h, y + h + padding_px)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            status_callback(f"    -> Skipped: Invalid crop dimensions after padding.\n")
            continue
        face_crop_bgr = original_image[crop_y1:crop_y2, crop_x1:crop_x2] # Crop from original for saving

        passed_skin_check = bypass_skin_tone or is_skin_tone(face_crop_bgr)
        if not passed_skin_check:
            status_callback(f"    -> Skipped: Failed skin-tone validation.\n")
            continue 

        # If validated, store crop and original box for drawing
        current_face_to_store = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2GRAY) if make_bw else face_crop_bgr
        valid_face_crops.append(current_face_to_store)
        validated_boxes_for_drawing.append((x, y, w, h)) # Original box, no padding
        status_callback(f"    -> Validated. Original box: [{x},{y},{w},{h}]\n")


    if not valid_face_crops and not (draw_on_original and validated_boxes_for_drawing):
        status_callback("No valid faces passed the verification process for cropping or drawing.\n")
        return

    base, ext = os.path.splitext(image_path)

    # --- Drawing on Original Image ---
    if draw_on_original and validated_boxes_for_drawing:
        for (x, y, w, h) in validated_boxes_for_drawing:
            if shape_type == 'rectangle':
                cv.rectangle(output_image_with_drawings, (x, y), (x+w, y+h), draw_color_bgr, 2)
            elif shape_type == 'circle':
                center_x = x + w // 2
                center_y = y + h // 2
                radius = max(w, h) // 2 # Simple radius, could be averaged or min/max
                cv.circle(output_image_with_drawings, (center_x, center_y), radius, draw_color_bgr, 2)
        
        drawn_output_path = f"{base}_detected_faces{ext}"
        cv.imwrite(drawn_output_path, output_image_with_drawings)
        status_callback(f"Saved image with detected faces drawn to: {drawn_output_path}\n")
        # If ONLY drawing is selected, we can return early if no other output is needed
        # For now, let's assume cropping/combining can still happen

    # --- Cropping and Combining Logic (if not drawing on original, or in addition to) ---
    # This part only runs if draw_on_original is FALSE OR if we want crops *in addition* to drawing.
    # For simplicity, let's assume if draw_on_original is true, we *don't* do separate crops/combine for now,
    # UNLESS we explicitly want both. The current logic will do both if valid_face_crops exist.
    # To make it exclusive:
    # if draw_on_original:
    #     status_callback("Drawing on original selected. Skipping individual crops/combine.\n")
    # else: # Only do this if not drawing on original
    #    ... (cropping/combining code below) ...
    # For now, let's allow both to happen if faces are validated.

    if valid_face_crops: # Only proceed if there are crops to save
        if combine_output:
            # ... (combine_output logic from previous version - remains the same) ...
            if not valid_face_crops: status_callback("No faces to combine.\n"); return # Should be redundant
            common_height = 150
            processed_for_concat = []
            for crop in valid_face_crops:
                h_crop, w_crop = crop.shape[:2]; scale = common_height / h_crop
                resized_crop = cv.resize(crop, (int(w_crop * scale), common_height))
                processed_for_concat.append(resized_crop)
            try:
                combined_image = cv.hconcat(processed_for_concat)
                output_path = f"{base}_combined_crops{ext}" # Renamed to avoid conflict
                cv.imwrite(output_path, combined_image)
                status_callback(f"Saved combined face crops to: {output_path}\n")
            except cv.error as e:
                status_callback(f"Error combining images: {e}\nSaving individual crops.\n")
                for i, crop_to_save in enumerate(valid_face_crops):
                    output_path = f"{base}_face_crop_{i+1}{ext}" # Renamed
                    cv.imwrite(output_path, crop_to_save)
                    status_callback(f"Saved: {output_path}\n")
        else: # Save individual crops
            for i, crop_to_save in enumerate(valid_face_crops):
                output_path = f"{base}_face_crop_{i+1}{ext}" # Renamed
                cv.imwrite(output_path, crop_to_save)
                status_callback(f"Saved: {output_path}\n")
    elif not draw_on_original: # If no crops and no drawing, means nothing was done
        status_callback("No valid faces to process or draw.\n")

    status_callback("Processing finished.\n")


# --- Tkinter UI ---
class FaceCropperApp:
    def __init__(self, root):
        self.root = root
        root.title("Advanced Face Detector & Cropper")
        root.geometry("780x750") # Adjusted size slightly

        style = ttk.Style()
        style.theme_use('clam') 
        style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        style.configure("TLabel", padding=3, font=('Helvetica', 10))
        style.configure("TCheckbutton", padding=3, font=('Helvetica', 10))
        style.configure("TRadiobutton", padding=3, font=('Helvetica', 10))
        style.configure("TEntry", padding=5, font=('Helvetica', 10))
        style.configure("TLabelframe.Label", font=('Helvetica', 11, 'bold'))
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'), padding=(0,10,0,5))

        main_paned_window = ttk.PanedWindow(root, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header_frame = ttk.Frame(main_paned_window, padding=10)
        main_paned_window.add(header_frame, weight=0) 

        app_title_label = ttk.Label(header_frame, text="Face Detector & Cropper Pro", style="Header.TLabel")
        app_title_label.pack(pady=(0, 5))

        purpose_text_area = tk.Text(header_frame, height=3, wrap=tk.WORD, relief=tk.FLAT,
                                     bg=root.cget('bg'), font=('Helvetica', 10), padx=5)
        purpose_text_area.insert(tk.END, 
            "Detect faces with MTCNN, adjust parameters, crop, or draw detections on the original image. "
            "Flexible options for preparing face datasets or visual analysis. "
            "An application created for Digital Image Processing - Final Project"
        )
        purpose_text_area.configure(state='disabled')
        purpose_text_area.pack(fill=tk.X, pady=(0, 10))

        controls_paned_window = ttk.PanedWindow(main_paned_window, orient=tk.HORIZONTAL)
        main_paned_window.add(controls_paned_window, weight=1)

        # --- Left Column of Controls ---
        left_column_frame = ttk.Frame(controls_paned_window, padding=5)
        controls_paned_window.add(left_column_frame, weight=1)

        file_frame = ttk.LabelFrame(left_column_frame, text="Image Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        self.file_label_var = tk.StringVar(value="No image selected")
        self.file_label = ttk.Label(file_frame, textvariable=self.file_label_var, wraplength=300)
        self.file_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10))
        self.select_button = ttk.Button(file_frame, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.RIGHT)

        basic_options_frame = ttk.LabelFrame(left_column_frame, text="Cropping & Output", padding=10)
        basic_options_frame.pack(fill=tk.X, pady=(0, 10))
        padding_frame = ttk.Frame(basic_options_frame)
        padding_frame.pack(fill=tk.X, pady=3)
        ttk.Label(padding_frame, text="Padding (px):").pack(side=tk.LEFT)
        self.padding_var = tk.IntVar(value=10)
        self.padding_entry = ttk.Entry(padding_frame, textvariable=self.padding_var, width=7, justify=tk.RIGHT)
        self.padding_entry.pack(side=tk.LEFT, padx=(5,0))
        self.combine_var = tk.BooleanVar(value=False)
        self.combine_check = ttk.Checkbutton(basic_options_frame, text="Combine Crops into One Image", variable=self.combine_var)
        self.combine_check.pack(anchor=tk.W, pady=3)
        self.bw_var = tk.BooleanVar(value=False)
        self.bw_check = ttk.Checkbutton(basic_options_frame, text="Convert Crops to B&W", variable=self.bw_var)
        self.bw_check.pack(anchor=tk.W, pady=3)

        # --- Right Column of Controls ---
        right_column_frame = ttk.Frame(controls_paned_window, padding=5)
        controls_paned_window.add(right_column_frame, weight=1)

        advanced_options_frame = ttk.LabelFrame(right_column_frame, text="Advanced Detection", padding=10)
        advanced_options_frame.pack(fill=tk.X, pady=(0, 10))
        confidence_frame = ttk.Frame(advanced_options_frame)
        confidence_frame.pack(fill=tk.X, pady=3)
        ttk.Label(confidence_frame, text="Confidence (0-1):").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.90)
        self.confidence_entry = ttk.Entry(confidence_frame, textvariable=self.confidence_var, width=7, justify=tk.RIGHT)
        self.confidence_entry.pack(side=tk.LEFT, padx=(5,0))
        self.bypass_keypoints_var = tk.BooleanVar(value=False)
        self.bypass_keypoints_check = ttk.Checkbutton(advanced_options_frame, text="Bypass Keypoint Validation", variable=self.bypass_keypoints_var)
        self.bypass_keypoints_check.pack(anchor=tk.W, pady=3)
        self.bypass_skin_tone_var = tk.BooleanVar(value=False)
        self.bypass_skin_tone_check = ttk.Checkbutton(advanced_options_frame, text="Bypass Skin Tone Validation", variable=self.bypass_skin_tone_var)
        self.bypass_skin_tone_check.pack(anchor=tk.W, pady=3)
        
        # --- Drawing Options (in the right column) ---
        drawing_options_frame = ttk.LabelFrame(right_column_frame, text="Drawing on Original", padding=10)
        drawing_options_frame.pack(fill=tk.X, pady=(10,0))

        self.draw_on_original_var = tk.BooleanVar(value=False)
        self.draw_on_original_check = ttk.Checkbutton(drawing_options_frame, text="Draw Detections on Original Image", 
                                                      variable=self.draw_on_original_var, command=self.toggle_drawing_options)
        self.draw_on_original_check.pack(anchor=tk.W, pady=3)

        self.shape_frame = ttk.Frame(drawing_options_frame) # Frame to hold shape and color
        # self.shape_frame.pack(fill=tk.X, padx=(20,0)) # Indent these options, initially hidden

        self.shape_type_var = tk.StringVar(value='rectangle')
        ttk.Label(self.shape_frame, text="Shape:").pack(side=tk.LEFT, pady=3)
        self.rect_radio = ttk.Radiobutton(self.shape_frame, text="Rectangle", variable=self.shape_type_var, value='rectangle')
        self.rect_radio.pack(side=tk.LEFT, padx=(5,5))
        self.circle_radio = ttk.Radiobutton(self.shape_frame, text="Circle", variable=self.shape_type_var, value='circle')
        self.circle_radio.pack(side=tk.LEFT)
        
        self.color_button = ttk.Button(self.shape_frame, text="Color", command=self.choose_draw_color, width=7)
        self.color_button.pack(side=tk.LEFT, padx=(10,0))
        self.draw_color_rgb = (0, 255, 0) # Default Green (RGB for color chooser)
        self.draw_color_bgr = (0, 255, 0) # Default Green (BGR for OpenCV)
        self.color_preview = tk.Label(self.shape_frame, text="  ", bg="#00FF00", width=2, relief=tk.SUNKEN)
        self.color_preview.pack(side=tk.LEFT, padx=(5,0))

        self.toggle_drawing_options() # Call once to set initial state

        # Process Button (centralized)
        process_button_frame = ttk.Frame(root, padding=(0,0,0,5)) # Add to root for centering under panes
        process_button_frame.pack(fill=tk.X)
        self.process_button = ttk.Button(process_button_frame, text="✨ Process Image ✨", command=self.process_image, state=tk.DISABLED)
        self.process_button.pack(pady=5) # pack in its own frame for centering

        log_frame = ttk.LabelFrame(main_paned_window, text="Processing Log", padding=10)
        main_paned_window.add(log_frame, weight=2) 
        self.status_text = scrolledtext.ScrolledText(log_frame, height=10, relief=tk.SUNKEN, borderwidth=1,
                                                     wrap=tk.WORD, font=('Consolas', 9) )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.status_text.configure(state='disabled')
        self.image_path = None

    def choose_draw_color(self):
        color_code = colorchooser.askcolor(title="Choose Detection Color", initialcolor=self.color_preview.cget("bg"))
        if color_code and color_code[0] is not None: # color_code is ((r,g,b), '#rrggbb')
            self.draw_color_rgb = tuple(int(c) for c in color_code[0]) # RGB for preview
            self.draw_color_bgr = (self.draw_color_rgb[2], self.draw_color_rgb[1], self.draw_color_rgb[0]) # BGR for OpenCV
            self.color_preview.config(bg=color_code[1]) # Hex for Tkinter bg

    def toggle_drawing_options(self):
        if self.draw_on_original_var.get():
            self.shape_frame.pack(fill=tk.X, padx=(20,0), pady=(0,5)) # Show and indent
        else:
            self.shape_frame.pack_forget() # Hide

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
            self.image_path = path; self.file_label_var.set(os.path.basename(path))
            self.process_button.config(state=tk.NORMAL)
            self._update_status(f"Selected: {path}\n")
        else:
            self.file_label_var.set("No image selected")
            self.process_button.config(state=tk.DISABLED)

    def process_image(self):
        if not self.image_path: messagebox.showerror("Error", "Please select an image first."); return
        if detector is None:
             messagebox.showerror("MTCNN Error", "MTCNN could not be initialized. Check console.")
             self._update_status("MTCNN not available. Aborting.\n"); return
        try:
            padding = self.padding_var.get()
            if padding < 0: messagebox.showerror("Error", "Padding cannot be negative."); return
            confidence_thresh = self.confidence_var.get()
            if not (0.0 <= confidence_thresh <= 1.0):
                messagebox.showerror("Error", "Confidence threshold must be between 0.0 and 1.0."); return
        except tk.TclError:
            messagebox.showerror("Error", "Invalid padding or confidence value."); return

        combine = self.combine_var.get()
        make_bw = self.bw_var.get()
        bypass_keypoints = self.bypass_keypoints_var.get()
        bypass_skin_tone = self.bypass_skin_tone_var.get()
        
        draw_on_orig = self.draw_on_original_var.get()
        shape = self.shape_type_var.get()
        # draw_color_bgr is already stored in self.draw_color_bgr

        self.process_button.config(state=tk.DISABLED)
        self._update_status("--- Starting processing ---\n")
        self._update_status(f"Settings: Padding={padding}px, CombineCrops={combine}, B&WCrops={make_bw}\n")
        self._update_status(f"Confidence: {confidence_thresh:.2f}, BypassKP={bypass_keypoints}, BypassSkin={bypass_skin_tone}\n")
        if draw_on_orig:
            self._update_status(f"Drawing on Original: Shape={shape}, Color(BGR)={self.draw_color_bgr}\n")
        
        try:
            detect_and_crop_faces(
                self.image_path, padding, combine, make_bw, 
                confidence_thresh, bypass_keypoints, bypass_skin_tone, 
                draw_on_orig, shape, self.draw_color_bgr, # Pass new drawing params
                self._update_status
            )
        except Exception as e:
            self._update_status(f"An unexpected error occurred: {e}\n")
            import traceback
            self._update_status(traceback.format_exc() + "\n")
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
        finally:
            self.process_button.config(state=tk.NORMAL)
            self._update_status("--- Processing complete ---\n\n")

if __name__ == "__main__":
    # Suppress TensorFlow messages (add os.environ line here if needed, before MTCNN import)
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    
    # Initialize MTCNN
    # (detector initialization and error handling as before)
    
    # --- Updated main block to handle MTCNN init failure better ---
    mtcnn_init_success = False
    try:
        # Re-check detector here in case it was already initialized globally
        # Or, better, move MTCNN init into the try block if not done already
        if detector is None: # If global init failed
             print("Attempting MTCNN initialization in __main__...")
             detector_local = MTCNN() # Try to init again
             detector = detector_local # Assign to global if successful
        mtcnn_init_success = True
    except Exception as e:
        print(f"Critical Error initializing MTCNN in __main__: {e}")
        # No need to set detector to None again if it's already None

    if not mtcnn_init_success or detector is None: # Double check
        root_temp = tk.Tk()
        root_temp.withdraw()
        messagebox.showerror("Startup Error", "MTCNN failed to initialize. The application cannot run.\nPlease check console for TensorFlow/PyTorch errors and ensure MTCNN is installed correctly.")
    else:
        root = tk.Tk()
        app = FaceCropperApp(root)
        root.mainloop()