import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps
import image_operations as img_ops

# Set modern customtkinter styling
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("UCS615: Desktop Image Processing Tool")
        self.geometry("1300x820")
        self.minsize(1100, 750)
        
        # State variables
        self.original_image = None  # PIL Image
        self.processed_image = None # PIL Image
        
        # Resized copies for display on canvas
        self.disp_orig_img = None
        self.disp_proc_img = None
        
        # Displays dimensions and scale ratios
        self.orig_scale_x = 1.0
        self.orig_scale_y = 1.0
        
        # Interaction modes
        self.crop_mode = False
        
        # Crop coordinate tracking
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect_id = None
        
        # Load a default test image to make app immediately functional
        self.load_default_image()
        
        # Create UI
        self.setup_ui()
        self.display_images()
        
    def load_default_image(self):
        # Generate a beautiful synthetic test image
        img = Image.new("RGB", (450, 450), color="white")
        draw = ImageDraw.Draw(img)
        # Background diagonal gradient
        for i in range(450):
            # RGB gradient
            r = int((i / 450.0) * 200)
            g = int(((450 - i) / 450.0) * 200)
            b = 150
            draw.line([(0, i), (450, i)], fill=(r, g, b))
        # Draw central circle
        draw.ellipse([150, 150, 300, 300], fill=(255, 215, 0), outline=(0, 0, 0), width=4)
        # Draw small dark square
        draw.rectangle([50, 50, 120, 120], fill=(30, 144, 255), outline=(0, 0, 0), width=3)
        # Draw a white rectangle (useful for showing thresholding/binary)
        draw.rectangle([320, 50, 400, 120], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        
        self.original_image = img
        self.processed_image = img.copy()
        
    def setup_ui(self):
        # 1. Main Grid Layout
        self.grid_rowconfigure(0, weight=0) # Sidebar top (fixed height)
        self.grid_rowconfigure(1, weight=1) # Sidebar scrollable + canvases
        self.grid_columnconfigure(0, weight=0) # Sidebar (fixed width)
        self.grid_columnconfigure(1, weight=1) # Visual display (resizable)
        
        # -------------------------------------------------------------
        # Sidebar Frame (Fixed Top)
        # -------------------------------------------------------------
        self.sidebar_fixed = ctk.CTkFrame(self, width=360, fg_color="transparent")
        self.sidebar_fixed.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        
        btn_frame = ctk.CTkFrame(self.sidebar_fixed, fg_color="transparent")
        btn_frame.pack(fill="x", padx=5, pady=2)
        
        self.btn_upload = ctk.CTkButton(btn_frame, text="Upload Image", command=self.upload_image, fg_color="#2ecc71", hover_color="#27ae60", height=32)
        self.btn_upload.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_reset = ctk.CTkButton(btn_frame, text="Reset Image", command=self.reset_image, fg_color="#e74c3c", hover_color="#c0392b", height=32)
        self.btn_reset.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # -------------------------------------------------------------
        # Sidebar Frame (Scrollable)
        # -------------------------------------------------------------
        self.sidebar = ctk.CTkScrollableFrame(self, width=360, label_text="Image Operations")
        self.sidebar.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        
        # --- Section 2: Point & Pixel Enhancements ---
        self.create_group_label(self.sidebar, "Spatial Point Processing")
        
        # Brightness Control
        ctk.CTkLabel(self.sidebar, text="Brightness").pack(anchor="w", padx=10)
        self.slider_bright = ctk.CTkSlider(self.sidebar, from_=0.0, to=2.0, number_of_steps=100, command=self.on_brightness_change)
        self.slider_bright.set(1.0)
        self.slider_bright.pack(fill="x", padx=10, pady=(0, 5))
        
        # Contrast Control
        ctk.CTkLabel(self.sidebar, text="Contrast").pack(anchor="w", padx=10)
        self.slider_contrast = ctk.CTkSlider(self.sidebar, from_=0.0, to=3.0, number_of_steps=100, command=self.on_contrast_change)
        self.slider_contrast.set(1.0)
        self.slider_contrast.pack(fill="x", padx=10, pady=(0, 5))
        
        # Single click transformations
        basics_frame1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        basics_frame1.pack(fill="x", padx=5, pady=5)
        
        self.btn_gray = ctk.CTkButton(basics_frame1, text="Grayscale (B&W)", command=self.apply_grayscale)
        self.btn_gray.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_neg = ctk.CTkButton(basics_frame1, text="Negative (Invert)", command=self.apply_negative)
        self.btn_neg.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        basics_frame2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        basics_frame2.pack(fill="x", padx=5, pady=2)
        
        self.btn_equalize = ctk.CTkButton(basics_frame2, text="Equalize Histogram", command=self.apply_histogram_equalization)
        self.btn_equalize.pack(fill="x", expand=True)
        
        # Thresholding
        self.create_sub_label(self.sidebar, "Binary Thresholding")
        thresh_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        thresh_frame.pack(fill="x", padx=5, pady=2)
        
        self.slider_thresh = ctk.CTkSlider(thresh_frame, from_=0, to=255, number_of_steps=255)
        self.slider_thresh.set(128)
        self.slider_thresh.pack(side="left", fill="x", expand=True, padx=(5, 5))
        
        self.btn_thresh = ctk.CTkButton(thresh_frame, text="Apply", width=60, command=self.apply_threshold_slider)
        self.btn_thresh.pack(side="right", padx=(0, 5))
        
        self.btn_otsu = ctk.CTkButton(self.sidebar, text="Otsu Auto Thresholding", command=self.apply_otsu)
        self.btn_otsu.pack(fill="x", padx=5, pady=2)
        
        # --- Section 3: Spatial Filtering ---
        self.create_group_label(self.sidebar, "Spatial Filters (Neighborhood)")
        
        # Blur controls
        self.create_sub_label(self.sidebar, "Image Smoothing (Blur)")
        blur_ctrl_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        blur_ctrl_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(blur_ctrl_frame, text="Kernel Size:").pack(side="left", padx=5)
        self.combo_blur_size = ctk.CTkComboBox(blur_ctrl_frame, values=["3", "5", "7", "9", "15"], width=70)
        self.combo_blur_size.set("5")
        self.combo_blur_size.pack(side="left", padx=5)
        
        blur_btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        blur_btn_frame.pack(fill="x", padx=5, pady=2)
        self.btn_box_blur = ctk.CTkButton(blur_btn_frame, text="Box Blur", command=self.apply_box_blur, width=90)
        self.btn_box_blur.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_gauss_blur = ctk.CTkButton(blur_btn_frame, text="Gaussian", command=self.apply_gaussian_blur, width=90)
        self.btn_gauss_blur.pack(side="left", fill="x", expand=True, padx=(3, 3))
        self.btn_median_filt = ctk.CTkButton(blur_btn_frame, text="Median", command=self.apply_median_filter, width=90)
        self.btn_median_filt.pack(side="left", fill="x", expand=True, padx=(3, 0))
        
        # Sharpen controls
        self.create_sub_label(self.sidebar, "Image Sharpening")
        sharp_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        sharp_frame.pack(fill="x", padx=5, pady=2)
        self.btn_sharp = ctk.CTkButton(sharp_frame, text="Laplacian Sharpen", command=self.apply_sharpen)
        self.btn_sharp.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_unsharp = ctk.CTkButton(sharp_frame, text="Unsharp Masking", command=self.apply_unsharp_mask)
        self.btn_unsharp.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # Edge Detection
        self.create_sub_label(self.sidebar, "Edge Detection")
        edges_frame1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        edges_frame1.pack(fill="x", padx=5, pady=2)
        self.btn_sobel = ctk.CTkButton(edges_frame1, text="Sobel", command=self.apply_sobel, width=100)
        self.btn_sobel.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_prewitt = ctk.CTkButton(edges_frame1, text="Prewitt", command=self.apply_prewitt, width=100)
        self.btn_prewitt.pack(side="left", fill="x", expand=True, padx=(3, 3))
        self.btn_canny = ctk.CTkButton(edges_frame1, text="Canny", command=self.apply_canny, width=100)
        self.btn_canny.pack(side="left", fill="x", expand=True, padx=(3, 0))
        
        edges_frame2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        edges_frame2.pack(fill="x", padx=5, pady=2)
        self.btn_log = ctk.CTkButton(edges_frame2, text="LoG (Laplacian of Gaussian)", command=self.apply_log)
        self.btn_log.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.btn_dog = ctk.CTkButton(edges_frame2, text="DoG (Diff of Gaussians)", command=self.apply_dog)
        self.btn_dog.pack(side="right", fill="x", expand=True, padx=(3, 0))
        
        # --- Section 4: Transforms ---
        self.create_group_label(self.sidebar, "Transforms & Mirroring")
        trans_frame1 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        trans_frame1.pack(fill="x", padx=5, pady=2)
        self.btn_mirror_h = ctk.CTkButton(trans_frame1, text="Mirror (Horiz)", command=lambda: self.apply_mirror("horizontal"))
        self.btn_mirror_h.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_mirror_v = ctk.CTkButton(trans_frame1, text="Mirror (Vert)", command=lambda: self.apply_mirror("vertical"))
        self.btn_mirror_v.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        trans_frame2 = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        trans_frame2.pack(fill="x", padx=5, pady=2)
        self.btn_rot_l = ctk.CTkButton(trans_frame2, text="Rotate 90 L", command=lambda: self.apply_rotate(90))
        self.btn_rot_l.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_rot_r = ctk.CTkButton(trans_frame2, text="Rotate 90 R", command=lambda: self.apply_rotate(-90))
        self.btn_rot_r.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # Interactive Crop Checkbox
        self.btn_crop_mode = ctk.CTkButton(self.sidebar, text="Interactive Crop: OFF", command=self.toggle_crop_mode, fg_color="#f39c12", hover_color="#d35400")
        self.btn_crop_mode.pack(fill="x", padx=5, pady=5)
        
        # --- Section 5: Color Spaces & Colormaps ---
        self.create_group_label(self.sidebar, "Color Processing")
        
        self.create_sub_label(self.sidebar, "Extract Color Model Channel")
        color_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        color_frame.pack(fill="x", padx=5, pady=2)
        self.combo_color_model = ctk.CTkComboBox(color_frame, values=["RGB", "HSV", "YCbCr"], width=90)
        self.combo_color_model.set("RGB")
        self.combo_color_model.pack(side="left", padx=5)
        
        self.combo_channel_idx = ctk.CTkComboBox(color_frame, values=["Channel 1", "Channel 2", "Channel 3"], width=100)
        self.combo_channel_idx.set("Channel 1")
        self.combo_channel_idx.pack(side="left", padx=5)
        
        self.btn_extract_channel = ctk.CTkButton(color_frame, text="Extract", width=60, command=self.apply_channel_extract)
        self.btn_extract_channel.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        self.create_sub_label(self.sidebar, "Pseudo-color Mapping")
        pseudo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        pseudo_frame.pack(fill="x", padx=5, pady=2)
        self.combo_cmap = ctk.CTkComboBox(pseudo_frame, values=["jet", "hot", "cool", "plasma", "inferno"], width=130)
        self.combo_cmap.set("jet")
        self.combo_cmap.pack(side="left", padx=5)
        
        self.btn_pseudo = ctk.CTkButton(pseudo_frame, text="Apply", command=self.apply_pseudocolor_map)
        self.btn_pseudo.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        self.btn_rgb_hist = ctk.CTkButton(self.sidebar, text="Show RGB Histogram", command=self.apply_rgb_histogram, fg_color="#34495e", hover_color="#2c3e50")
        self.btn_rgb_hist.pack(fill="x", padx=5, pady=5)
        
        # --- Section 6: Frequency Domain (DFT/FFT) ---
        self.create_group_label(self.sidebar, "Frequency Domain (DFT)")
        
        # DFT Spectrum Visualizer
        self.btn_fft_spectrum = ctk.CTkButton(self.sidebar, text="Show Log FFT Magnitude Spectrum", command=self.apply_fft_spectrum)
        self.btn_fft_spectrum.pack(fill="x", padx=5, pady=2)
        
        # Frequency domain LPF/HPF
        self.create_sub_label(self.sidebar, "Fourier Domain Filters")
        freq_filt_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        freq_filt_frame.pack(fill="x", padx=5, pady=2)
        self.combo_freq_filt = ctk.CTkComboBox(freq_filt_frame, values=["ideal_lpf", "ideal_hpf", "gaussian_lpf", "gaussian_hpf"], width=130)
        self.combo_freq_filt.set("gaussian_lpf")
        self.combo_freq_filt.pack(side="left", padx=5)
        
        self.btn_apply_freq = ctk.CTkButton(freq_filt_frame, text="Filter", command=self.apply_freq_filter)
        self.btn_apply_freq.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        ctk.CTkLabel(self.sidebar, text="Filter Cutoff Radius").pack(anchor="w", padx=10)
        self.slider_cutoff = ctk.CTkSlider(self.sidebar, from_=5, to=150, number_of_steps=145)
        self.slider_cutoff.set(30)
        self.slider_cutoff.pack(fill="x", padx=10, pady=(0, 5))
        

        
        # --- Section 7: Feature Extraction ---
        self.create_group_label(self.sidebar, "Feature Extraction")
        
        feat_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        feat_frame.pack(fill="x", padx=5, pady=2)
        self.btn_harris = ctk.CTkButton(feat_frame, text="Harris Corners", command=self.apply_harris)
        self.btn_harris.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_lbp = ctk.CTkButton(feat_frame, text="Local Binary Patterns (LBP)", command=self.apply_lbp)
        self.btn_lbp.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # --- Section 8: Image Segmentation ---
        self.create_group_label(self.sidebar, "Image Segmentation")
        
        # K-Means Clustering
        self.create_sub_label(self.sidebar, "K-Means Clustering Segmentation")
        kmeans_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        kmeans_frame.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(kmeans_frame, text="Clusters K:").pack(side="left", padx=5)
        self.slider_kmeans_k = ctk.CTkSlider(kmeans_frame, from_=2, to=8, number_of_steps=6, width=120)
        self.slider_kmeans_k.set(3)
        self.slider_kmeans_k.pack(side="left", padx=5)
        self.btn_kmeans = ctk.CTkButton(kmeans_frame, text="Run", width=60, command=self.apply_kmeans)
        self.btn_kmeans.pack(side="right", padx=5)
        

        
        # Morphology (Binary Operations)
        self.create_sub_label(self.sidebar, "Binary Morphological Operations")
        morph_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        morph_frame.pack(fill="x", padx=5, pady=2)
        self.combo_morph_op = ctk.CTkComboBox(morph_frame, values=["dilation", "erosion", "opening", "closing"], width=130)
        self.combo_morph_op.set("dilation")
        self.combo_morph_op.pack(side="left", padx=5)
        
        self.btn_morph = ctk.CTkButton(morph_frame, text="Apply", command=self.apply_morphology)
        self.btn_morph.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # --- Section 9: Security & Compression (Applications) ---
        self.create_group_label(self.sidebar, "Applications (Compression & Security)")
        
        # LSB Steganography
        self.create_sub_label(self.sidebar, "LSB Steganography (Watermarking)")
        self.txt_steg_msg = ctk.CTkEntry(self.sidebar, placeholder_text="Enter secret message to hide...")
        self.txt_steg_msg.pack(fill="x", padx=10, pady=2)
        
        steg_btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        steg_btn_frame.pack(fill="x", padx=5, pady=2)
        self.btn_steg_enc = ctk.CTkButton(steg_btn_frame, text="Encode Message", command=self.apply_steg_encode)
        self.btn_steg_enc.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_steg_dec = ctk.CTkButton(steg_btn_frame, text="Decode Message", command=self.apply_steg_decode)
        self.btn_steg_dec.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # Block-DCT compression
        self.create_sub_label(self.sidebar, "Block DCT Compression Simulation")
        ctk.CTkLabel(self.sidebar, text="Quality Parameter").pack(anchor="w", padx=10)
        self.slider_dct_quality = ctk.CTkSlider(self.sidebar, from_=5, to=100, number_of_steps=95)
        self.slider_dct_quality.set(50)
        self.slider_dct_quality.pack(fill="x", padx=10, pady=(0, 5))
        self.btn_dct = ctk.CTkButton(self.sidebar, text="Apply Block DCT Compression", command=self.apply_dct_compression)
        self.btn_dct.pack(fill="x", padx=5, pady=2)
        
        # -------------------------------------------------------------
        # Visual Viewport (Right Panel)
        # -------------------------------------------------------------
        self.main_view = ctk.CTkFrame(self)
        self.main_view.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.main_view.grid_rowconfigure(0, weight=1)
        self.main_view.grid_columnconfigure(0, weight=1) # Left Image (Original)
        self.main_view.grid_columnconfigure(1, weight=1) # Right Image (Processed)
        
        # Left Panel (Original)
        self.panel_orig_frame = ctk.CTkFrame(self.main_view)
        self.panel_orig_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.panel_orig_frame.grid_rowconfigure(0, weight=0)
        self.panel_orig_frame.grid_rowconfigure(1, weight=1)
        self.panel_orig_frame.grid_columnconfigure(0, weight=1)
        
        lbl_orig_title = ctk.CTkLabel(self.panel_orig_frame, text="Original Image", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_orig_title.grid(row=0, column=0, pady=5, sticky="ew")
        
        self.canvas_orig = tk.Canvas(self.panel_orig_frame, bg="#2c3e50", highlightthickness=0)
        self.canvas_orig.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Right Panel (Processed)
        self.panel_proc_frame = ctk.CTkFrame(self.main_view)
        self.panel_proc_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.panel_proc_frame.grid_rowconfigure(0, weight=0)
        self.panel_proc_frame.grid_rowconfigure(1, weight=1)
        self.panel_proc_frame.grid_columnconfigure(0, weight=1)
        
        lbl_proc_title = ctk.CTkLabel(self.panel_proc_frame, text="Processed Image", font=ctk.CTkFont(size=16, weight="bold"))
        lbl_proc_title.grid(row=0, column=0, pady=5, sticky="ew")
        
        self.canvas_proc = tk.Canvas(self.panel_proc_frame, bg="#2c3e50", highlightthickness=0)
        self.canvas_proc.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Bind canvas interaction events
        self.canvas_proc.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas_proc.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas_proc.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Bind window resize event to redraw images correctly
        self.bind("<Configure>", self.on_window_resize)
        
        # Status Bar
        self.lbl_status = ctk.CTkLabel(self, text="Status: Ready. Load an image or modify parameters.", font=ctk.CTkFont(size=13), anchor="w", fg_color="#34495e", height=28)
        self.lbl_status.grid(row=2, column=0, columnspan=2, sticky="ew")

    # Helper label factories
    def create_group_label(self, parent, text):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold"), text_color="#3498db")
        lbl.pack(anchor="w", padx=5, pady=(15, 2))
        
    def create_sub_label(self, parent, text):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=11, slant="italic"), text_color="#bdc3c7")
        lbl.pack(anchor="w", padx=10, pady=(5, 1))

    # -------------------------------------------------------------
    # Image Operations Pipeline
    # -------------------------------------------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")]
        )
        if file_path:
            try:
                # Load image via Pillow, transpose based on EXIF tags, and ensure standard RGB mode
                raw_img = Image.open(file_path)
                img = ImageOps.exif_transpose(raw_img).convert("RGB")
                # Downsize if extremely large (to prevent UI lag in Python)
                if max(img.size) > 1000:
                    img.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                    
                self.original_image = img
                self.processed_image = img.copy()
                
                # Reset sliders to default values
                self.slider_bright.set(1.0)
                self.slider_contrast.set(1.0)
                self.slider_thresh.set(128)
                
                self.set_status(f"Loaded image: {file_path.split('/')[-1]} | Resolution: {img.width}x{img.height}")
                self.display_images()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def reset_image(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            # Reset UI settings
            self.slider_bright.set(1.0)
            self.slider_contrast.set(1.0)
            self.slider_thresh.set(128)
            self.deactivate_crop()
            self.display_images()
            self.set_status("Reverted to original image.")
            
    def display_images(self, orig_override=None):
        if not self.original_image:
            return
            
        # Draw on Original Canvas
        img_to_show_orig = orig_override if orig_override is not None else self.original_image
        orig_w, orig_h = img_to_show_orig.size
        cw_orig = self.canvas_orig.winfo_width()
        ch_orig = self.canvas_orig.winfo_height()
        
        # Handle configuration issues when window is drawing initially
        if cw_orig <= 1: cw_orig = 450
        if ch_orig <= 1: ch_orig = 450
        
        # Calculate aspect ratio scale
        scale = min(cw_orig / orig_w, ch_orig / orig_h)
        disp_w = int(orig_w * scale)
        disp_h = int(orig_h * scale)
        
        # Scale for coordinate conversion (original image coordinates vs displayed canvas coordinates)
        self.orig_scale_x = orig_w / disp_w
        self.orig_scale_y = orig_h / disp_h
        
        # Resize original image for canvas preview
        self.disp_orig_img = img_to_show_orig.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self.photo_orig = ImageTk.PhotoImage(self.disp_orig_img)
        
        self.canvas_orig.delete("all")
        # Center image in canvas
        self.canvas_orig.create_image(cw_orig // 2, ch_orig // 2, image=self.photo_orig, anchor="center")
        
        # Draw on Processed Canvas
        proc_w, proc_h = self.processed_image.size
        cw_proc = self.canvas_proc.winfo_width()
        ch_proc = self.canvas_proc.winfo_height()
        
        if cw_proc <= 1: cw_proc = 450
        if ch_proc <= 1: ch_proc = 450
        
        scale_proc = min(cw_proc / proc_w, ch_proc / proc_h)
        disp_proc_w = int(proc_w * scale_proc)
        disp_proc_h = int(proc_h * scale_proc)
        
        # Center coordinates offset for coordinate click mapping on Processed Image
        self.proc_disp_offset_x = (cw_proc - disp_proc_w) // 2
        self.proc_disp_offset_y = (ch_proc - disp_proc_h) // 2
        self.proc_disp_w = disp_proc_w
        self.proc_disp_h = disp_proc_h
        
        # Scale factors specifically for the processed canvas coordinate conversion
        self.proc_scale_x = proc_w / disp_proc_w
        self.proc_scale_y = proc_h / disp_proc_h
        
        # Resize processed image for canvas preview
        self.disp_proc_img = self.processed_image.resize((disp_proc_w, disp_proc_h), Image.Resampling.BILINEAR)
        self.photo_proc = ImageTk.PhotoImage(self.disp_proc_img)
        
        self.canvas_proc.delete("all")
        self.canvas_proc.create_image(cw_proc // 2, ch_proc // 2, image=self.photo_proc, anchor="center")
        
    def on_window_resize(self, event):
        # Trigger redraw only on significant size changes
        if event.widget == self:
            self.display_images()

    def set_status(self, text):
        self.lbl_status.configure(text=f" Status: {text}")

    # -------------------------------------------------------------
    # Action Handlers: Basic Adjustments
    # -------------------------------------------------------------
    def on_brightness_change(self, val):
        if self.original_image:
            # We chain bright & contrast off original image to simulate sliders
            # Get current contrast slider val
            contrast_val = self.slider_contrast.get()
            np_orig = img_ops.pil_to_numpy(self.original_image)
            temp = img_ops.adjust_brightness(np_orig, val)
            temp = img_ops.adjust_contrast(temp, contrast_val)
            self.processed_image = img_ops.numpy_to_pil(temp)
            self.display_images()
            self.set_status(f"Adjusted brightness to {val:.2f}")

    def on_contrast_change(self, val):
        if self.original_image:
            bright_val = self.slider_bright.get()
            np_orig = img_ops.pil_to_numpy(self.original_image)
            temp = img_ops.adjust_brightness(np_orig, bright_val)
            temp = img_ops.adjust_contrast(temp, val)
            self.processed_image = img_ops.numpy_to_pil(temp)
            self.display_images()
            self.set_status(f"Adjusted contrast to {val:.2f}")

    def apply_grayscale(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        gray = img_ops.convert_to_grayscale(np_proc)
        self.processed_image = img_ops.numpy_to_pil(gray)
        self.display_images()
        self.set_status("Converted workspace to grayscale.")

    def apply_negative(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        neg = img_ops.invert_image(np_proc)
        self.processed_image = img_ops.numpy_to_pil(neg)
        self.display_images()
        self.set_status("Applied image negative.")

    def apply_histogram_equalization(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        equalized = img_ops.equalize_histogram(np_proc)
        self.processed_image = img_ops.numpy_to_pil(equalized)
        self.display_images()
        self.set_status("Applied color-preserving Histogram Equalization.")

    def apply_threshold_slider(self):
        thresh_val = int(self.slider_thresh.get())
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        binary = img_ops.apply_threshold(np_proc, thresh_val)
        self.processed_image = img_ops.numpy_to_pil(binary)
        self.display_images()
        self.set_status(f"Applied manual threshold (value: {thresh_val}).")
        
    def apply_otsu(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        binary, thresh_val = img_ops.otsu_threshold(np_proc)
        self.slider_thresh.set(thresh_val)
        self.processed_image = img_ops.numpy_to_pil(binary)
        self.display_images()
        self.set_status(f"Otsu's Threshold computed: {thresh_val}. Applied binary threshold.")

    # -------------------------------------------------------------
    # Action Handlers: Spatial Filters
    # -------------------------------------------------------------
    def get_blur_size(self):
        return int(self.combo_blur_size.get())

    def apply_box_blur(self):
        size = self.get_blur_size()
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        blurred = img_ops.box_blur(np_proc, size)
        self.processed_image = img_ops.numpy_to_pil(blurred)
        self.display_images()
        self.set_status(f"Applied Box Blur ({size}x{size}).")

    def apply_gaussian_blur(self):
        size = self.get_blur_size()
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        # Choose reasonable sigma
        sigma = size / 3.0
        blurred = img_ops.gaussian_blur(np_proc, size, sigma)
        self.processed_image = img_ops.numpy_to_pil(blurred)
        self.display_images()
        self.set_status(f"Applied Gaussian Blur ({size}x{size}, sigma: {sigma:.2f}).")

    def apply_median_filter(self):
        size = self.get_blur_size()
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        median_filtered = img_ops.median_filter(np_proc, size)
        self.processed_image = img_ops.numpy_to_pil(median_filtered)
        self.display_images()
        self.set_status(f"Applied Median Filter ({size}x{size}).")

    def apply_sharpen(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        sharpened = img_ops.laplacian_sharpen(np_proc, strength=1.0)
        self.processed_image = img_ops.numpy_to_pil(sharpened)
        self.display_images()
        self.set_status("Applied Laplacian spatial sharpening.")

    def apply_unsharp_mask(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        sharpened = img_ops.unsharp_mask(np_proc, blur_size=5, strength=1.5)
        self.processed_image = img_ops.numpy_to_pil(sharpened)
        self.display_images()
        self.set_status("Applied Unsharp Masking filter.")

    def apply_sobel(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        edges = img_ops.edge_sobel(np_proc)
        self.processed_image = img_ops.numpy_to_pil(edges)
        self.display_images()
        self.set_status("Applied Sobel edge detection.")

    def apply_prewitt(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        edges = img_ops.edge_prewitt(np_proc)
        self.processed_image = img_ops.numpy_to_pil(edges)
        self.display_images()
        self.set_status("Applied Prewitt edge detection.")

    def apply_canny(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        edges = img_ops.edge_canny(np_proc, low_threshold=30, high_threshold=80)
        self.processed_image = img_ops.numpy_to_pil(edges)
        self.display_images()
        self.set_status("Applied Canny edge detection (low=30, high=80).")

    def apply_log(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        edges = img_ops.edge_log(np_proc, size=9, sigma=1.4)
        self.processed_image = img_ops.numpy_to_pil(edges)
        self.display_images()
        self.set_status("Applied Laplacian of Gaussian (LoG) edge detection.")

    def apply_dog(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        edges = img_ops.edge_dog(np_proc, sigma1=1.0, sigma2=2.0)
        self.processed_image = img_ops.numpy_to_pil(edges)
        self.display_images()
        self.set_status("Applied Difference of Gaussians (DoG) edge detection.")

    # -------------------------------------------------------------
    # Action Handlers: Transforms & Mirroring
    # -------------------------------------------------------------
    def apply_mirror(self, direction):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        mirrored = img_ops.mirror_image(np_proc, direction)
        self.processed_image = img_ops.numpy_to_pil(mirrored)
        self.display_images()
        self.set_status(f"Flipped image {direction}ly.")

    def apply_rotate(self, angle):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        rotated = img_ops.rotate_image(np_proc, angle)
        self.processed_image = img_ops.numpy_to_pil(rotated)
        self.display_images()
        self.set_status(f"Rotated image by {-angle} degrees.")

    def toggle_crop_mode(self):
        if self.crop_mode:
            self.deactivate_crop()
        else:
            self.crop_mode = True
            self.btn_crop_mode.configure(text="Interactive Crop: ON", fg_color="#2ecc71", hover_color="#27ae60")
            self.set_status("Crop Mode Active. Click & drag a crop box on the processed canvas.")

    def deactivate_crop(self):
        self.crop_mode = False
        self.btn_crop_mode.configure(text="Interactive Crop: OFF", fg_color="#f39c12", hover_color="#d35400")
        if self.crop_rect_id:
            self.canvas_proc.delete(self.crop_rect_id)
            self.crop_rect_id = None

    # -------------------------------------------------------------
    # Action Handlers: Color Space Processing
    # -------------------------------------------------------------
    def apply_channel_extract(self):
        model = self.combo_color_model.get()
        channel_name = self.combo_channel_idx.get()
        # Map channel name to index
        channel_idx = ["Channel 1", "Channel 2", "Channel 3"].index(channel_name)
        
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        channels = img_ops.get_color_channels(np_proc, model)
        
        # Display selected channel
        self.processed_image = img_ops.numpy_to_pil(channels[channel_idx])
        self.display_images()
        self.set_status(f"Extracted {model} - {channel_name}.")

    def apply_pseudocolor_map(self):
        cmap_name = self.combo_cmap.get()
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        colored = img_ops.apply_pseudocolor(np_proc, cmap_name)
        self.processed_image = img_ops.numpy_to_pil(colored)
        self.display_images()
        self.set_status(f"Applied pseudocolor mapping using '{cmap_name}' colormap.")

    def apply_rgb_histogram(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        hist_img = img_ops.get_rgb_histogram(np_proc)
        self.processed_image = hist_img
        self.display_images()
        self.set_status("Displayed real-time RGB histogram in the processed image canvas.")

    # -------------------------------------------------------------
    # Action Handlers: Frequency Domain
    # -------------------------------------------------------------
    def apply_fft_spectrum(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        f_shift = img_ops.compute_fft(np_proc)
        mag_spec = img_ops.get_fft_magnitude_spectrum(f_shift)
        
        # Render magnitude spectrum as RGB
        self.processed_image = img_ops.numpy_to_pil(mag_spec).convert("RGB")
        self.display_images()
        self.set_status("Displayed 2D Fourier Magnitude Spectrum.")

    def apply_freq_filter(self):
        filt_type = self.combo_freq_filt.get()
        cutoff = float(self.slider_cutoff.get())
        
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        reconstructed, spectrum, mask_img = img_ops.apply_frequency_filter(np_proc, filt_type, cutoff)
        
        # Set processed image to filter output
        self.processed_image = img_ops.numpy_to_pil(reconstructed)
        self.display_images()
        self.set_status(f"Applied Fourier Filter: {filt_type} (cutoff: {cutoff:.1f}).")

    # -------------------------------------------------------------
    # Action Handlers: Feature Extraction
    # -------------------------------------------------------------
    def apply_harris(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        output = img_ops.harris_corner_detector(np_proc)
        self.processed_image = img_ops.numpy_to_pil(output)
        self.display_images()
        self.set_status("Harris Corners highlighted in red.")

    def apply_lbp(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        lbp = img_ops.compute_lbp(np_proc)
        self.processed_image = img_ops.numpy_to_pil(lbp)
        self.display_images()
        self.set_status("Computed Local Binary Patterns (LBP) texture representation.")

    # -------------------------------------------------------------
    # Action Handlers: Image Segmentation
    # -------------------------------------------------------------
    def apply_kmeans(self):
        k = int(self.slider_kmeans_k.get())
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        self.set_status(f"Running K-Means segmentation (K={k})... Please wait.")
        self.update_idletasks() # Refresh UI
        
        segmented = img_ops.kmeans_segmentation(np_proc, k)
        self.processed_image = img_ops.numpy_to_pil(segmented)
        self.display_images()
        self.set_status(f"K-Means segmentation complete with {k} clusters.")

    def apply_morphology(self):
        op_type = self.combo_morph_op.get()
        thresh_val = int(self.slider_thresh.get())
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        
        morph = img_ops.binary_morphology(np_proc, op_type, thresh_val)
        self.processed_image = img_ops.numpy_to_pil(morph)
        self.display_images()
        self.set_status(f"Applied binary morphology: {op_type} (binarized at threshold {thresh_val}).")

    # -------------------------------------------------------------
    # Action Handlers: Compression & Security
    # -------------------------------------------------------------
    def apply_steg_encode(self):
        msg = self.txt_steg_msg.get()
        if not msg:
            messagebox.showwarning("Warning", "Please enter a message in the text entry to encode.")
            return
            
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        try:
            encoded = img_ops.lsb_steganography_encode(np_proc, msg)
            self.processed_image = img_ops.numpy_to_pil(encoded)
            self.display_images()
            self.set_status(f"Encoded '{msg}' in Red channel LSB.")
            self.txt_steg_msg.delete(0, tk.END)
        except ValueError as ve:
            messagebox.showerror("Error", str(ve))

    def apply_steg_decode(self):
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        msg = img_ops.lsb_steganography_decode(np_proc)
        if msg:
            self.set_status(f"Decoded hidden LSB message: \"{msg}\"")
            messagebox.showinfo("LSB Decoded Message", f"Found Message:\n\n{msg}")
        else:
            self.set_status("No hidden LSB steganography message detected.")
            messagebox.showinfo("LSB Decoded Message", "No hidden steganography message found in Red channel LSB.")

    def apply_dct_compression(self):
        quality = int(self.slider_dct_quality.get())
        np_proc = img_ops.pil_to_numpy(self.processed_image)
        
        compressed, mse, psnr = img_ops.block_dct_compression(np_proc, quality)
        self.processed_image = img_ops.numpy_to_pil(compressed)
        self.display_images()
        self.set_status(f"Block-DCT Quantization (quality: {quality}): MSE={mse:.2f}, PSNR={psnr:.2f}dB.")

    # -------------------------------------------------------------
    # Canvas Interactive Handling (Crop & Region Growing)
    # -------------------------------------------------------------
    def get_clicked_pixel_coords(self, canvas_x, canvas_y):
        # 1. Translate canvas coordinate to image display boundary coordinate
        x_in_disp = canvas_x - self.proc_disp_offset_x
        y_in_disp = canvas_y - self.proc_disp_offset_y
        
        # 2. Check if inside display box bounds
        if 0 <= x_in_disp < self.proc_disp_w and 0 <= y_in_disp < self.proc_disp_h:
            # 3. Translate display box coordinate to actual image resolution coordinate
            img_x = int(x_in_disp * self.proc_scale_x)
            img_y = int(y_in_disp * self.proc_scale_y)
            return img_x, img_y, x_in_disp, y_in_disp
            
        return None

    def on_canvas_press(self, event):
        if self.crop_mode:
            coords = self.get_clicked_pixel_coords(event.x, event.y)
            if coords:
                img_x, img_y, disp_x, disp_y = coords
                # Set initial crop start point (in canvas coordinate system)
                self.crop_start_x = event.x
                self.crop_start_y = event.y
                
                # Delete existing selection box
                if self.crop_rect_id:
                    self.canvas_proc.delete(self.crop_rect_id)
                self.crop_rect_id = self.canvas_proc.create_rectangle(
                    self.crop_start_x, self.crop_start_y, self.crop_start_x, self.crop_start_y,
                    outline="yellow", width=2, dash=(4, 4)
                )
    def on_canvas_drag(self, event):
        if self.crop_mode and self.crop_start_x is not None:
            # Update dashed box coordinates as cursor drags
            self.canvas_proc.coords(self.crop_rect_id, self.crop_start_x, self.crop_start_y, event.x, event.y)

    def on_canvas_release(self, event):
        if self.crop_mode and self.crop_start_x is not None:
            # Map start and end coordinates back to image pixel space
            start_coords = self.get_clicked_pixel_coords(self.crop_start_x, self.crop_start_y)
            end_coords = self.get_clicked_pixel_coords(event.x, event.y)
            
            if start_coords and end_coords:
                x1, y1 = start_coords[0], start_coords[1]
                x2, y2 = end_coords[0], end_coords[1]
                
                # Ensure ordered coordinates
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Execute Crop
                np_proc = img_ops.pil_to_numpy(self.processed_image)
                cropped = img_ops.crop_image(np_proc, x1, y1, x2, y2)
                self.processed_image = img_ops.numpy_to_pil(cropped)
                
                # Redraw
                self.deactivate_crop()
                self.display_images()
                self.set_status(f"Cropped image to box: ({x1}, {y1}) to ({x2}, {y2}). Resolution: {cropped.shape[1]}x{cropped.shape[0]}.")
            else:
                self.deactivate_crop()
                self.set_status("Crop selection out of bounds.")
                
            self.crop_start_x = None
            self.crop_start_y = None

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
