import streamlit as st
import numpy as np
from PIL import Image
import image_operations as img_ops
import io

# Page configuration
st.set_page_config(
    page_title="UCS615: Image Processing Web Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("UCS615: Digital Image Processing Web Tool")
st.markdown("A web-based image processing workbench implementing spatial domain point processing, neighborhood filtering, color models, frequency transformations (DFT), segmentation, steganography, and compression.")

# ---------------------------------------------------------
# Sidebar - File Actions & Parameters
# ---------------------------------------------------------
st.sidebar.header("File Actions")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp"])

# State management for processed image
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Default test pattern generator if no image is uploaded
def load_default_image():
    img = Image.new("RGB", (450, 450), color="white")
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img)
    for i in range(450):
        r = int((i / 450.0) * 200)
        g = int(((450 - i) / 450.0) * 200)
        b = 150
        draw.line([(0, i), (450, i)], fill=(r, g, b))
    draw.ellipse([150, 150, 300, 300], fill=(255, 215, 0), outline=(0, 0, 0), width=4)
    draw.rectangle([50, 50, 120, 120], fill=(30, 144, 255), outline=(0, 0, 0), width=3)
    draw.rectangle([320, 50, 400, 120], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    return img

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.session_state.original_image = img
    # Only overwrite processed image if it's a new upload
    if st.sidebar.button("Reload Original Upload"):
        st.session_state.processed_image = img.copy()
        st.success("Reloaded original image!")
else:
    if st.session_state.original_image is None:
        st.session_state.original_image = load_default_image()
        st.session_state.processed_image = st.session_state.original_image.copy()

# Ensure processed image is initialized
if st.session_state.processed_image is None:
    st.session_state.processed_image = st.session_state.original_image.copy()

# Add a Reset Button in sidebar
if st.sidebar.button("Reset Processed Image"):
    st.session_state.processed_image = st.session_state.original_image.copy()
    st.success("Reset all operations!")

# Helper to run operations and update state
def apply_op(op_func, *args, **kwargs):
    np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
    res_np = op_func(np_proc, *args, **kwargs)
    st.session_state.processed_image = img_ops.numpy_to_pil(res_np)

# ---------------------------------------------------------
# Sidebar - Operations
# ---------------------------------------------------------
st.sidebar.header("Operations")

# Category 1: Point Processing
with st.sidebar.expander("Spatial Point Processing"):
    # Brightness & Contrast
    bright_factor = st.slider("Brightness", 0.0, 2.0, 1.0, 0.05)
    contrast_factor = st.slider("Contrast", 0.0, 3.0, 1.0, 0.05)
    
    if st.button("Apply Brightness & Contrast"):
        np_orig = img_ops.pil_to_numpy(st.session_state.original_image)
        temp = img_ops.adjust_brightness(np_orig, bright_factor)
        temp = img_ops.adjust_contrast(temp, contrast_factor)
        st.session_state.processed_image = img_ops.numpy_to_pil(temp)
        st.success("Applied Brightness & Contrast!")

    st.markdown("---")
    
    if st.button("Grayscale (B&W)"):
        apply_op(img_ops.convert_to_grayscale)
        st.success("Converted to Grayscale!")
        
    if st.button("Negative (Invert)"):
        apply_op(img_ops.invert_image)
        st.success("Inverted Image!")
        
    if st.button("Equalize Histogram"):
        apply_op(img_ops.equalize_histogram)
        st.success("Equalized Histogram!")
        
    st.markdown("---")
    # Manual thresholding
    thresh_val = st.slider("Threshold Value", 0, 255, 128)
    if st.button("Apply Threshold"):
        apply_op(img_ops.apply_threshold, thresh_val)
        st.success(f"Applied threshold at {thresh_val}!")
        
    if st.button("Otsu's Auto Thresholding"):
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        binary, t_val = img_ops.otsu_threshold(np_proc)
        st.session_state.processed_image = img_ops.numpy_to_pil(binary)
        st.success(f"Otsu's Threshold calculated: {t_val}. Applied binary threshold!")

# Category 2: Spatial Neighborhood Filters
with st.sidebar.expander("Spatial Filtering & Smoothing"):
    kernel_size = st.selectbox("Kernel Size", [3, 5, 7, 9, 15], index=1)
    
    if st.button("Box Blur"):
        apply_op(img_ops.box_blur, kernel_size)
        st.success("Applied Box Blur!")
        
    if st.button("Gaussian Blur"):
        apply_op(img_ops.gaussian_blur, kernel_size, kernel_size / 3.0)
        st.success("Applied Gaussian Blur!")
        
    if st.button("Median Filter (Denoise)"):
        apply_op(img_ops.median_filter, kernel_size)
        st.success("Applied Median Filter!")

    st.markdown("---")
    
    if st.button("Laplacian Sharpen"):
        apply_op(img_ops.laplacian_sharpen)
        st.success("Applied Laplacian Sharpening!")
        
    if st.button("Unsharp Masking"):
        apply_op(img_ops.unsharp_mask)
        st.success("Applied Unsharp Masking!")

# Category 3: Edge Detection
with st.sidebar.expander("Edge Detection"):
    if st.button("Sobel Edge Detection"):
        apply_op(img_ops.edge_sobel)
        st.success("Applied Sobel Edge Detection!")
        
    if st.button("Prewitt Edge Detection"):
        apply_op(img_ops.edge_prewitt)
        st.success("Applied Prewitt Edge Detection!")
        
    if st.button("Canny Edge Detection"):
        apply_op(img_ops.edge_canny)
        st.success("Applied Canny Edge Detection!")
        
    if st.button("Laplacian of Gaussian (LoG)"):
        apply_op(img_ops.edge_log)
        st.success("Applied LoG Edge Detection!")
        
    if st.button("Difference of Gaussians (DoG)"):
        apply_op(img_ops.edge_dog)
        st.success("Applied DoG Edge Detection!")

# Category 4: Transforms
with st.sidebar.expander("Transforms & Mirroring"):
    if st.button("Mirror Horizontal"):
        apply_op(img_ops.mirror_image, "horizontal")
        st.success("Mirrored Horizontally!")
        
    if st.button("Mirror Vertical"):
        apply_op(img_ops.mirror_image, "vertical")
        st.success("Mirrored Vertically!")
        
    if st.button("Rotate 90° Left"):
        apply_op(img_ops.rotate_image, 90)
        st.success("Rotated Left!")
        
    if st.button("Rotate 90° Right"):
        apply_op(img_ops.rotate_image, -90)
        st.success("Rotated Right!")
        
    st.markdown("---")
    # Interactive cropping via sliders (Streamlit simple crop)
    w_curr, h_curr = st.session_state.processed_image.size
    crop_x = st.slider("Crop Horizontal Bounds", 0, w_curr, (0, w_curr))
    crop_y = st.slider("Crop Vertical Bounds", 0, h_curr, (0, h_curr))
    
    if st.button("Apply Crop Box"):
        apply_op(img_ops.crop_image, crop_x[0], crop_y[0], crop_x[1], crop_y[1])
        st.success("Cropped Image!")

# Category 5: Color Spaces & Channels
with st.sidebar.expander("Color Spaces & Colormaps"):
    model = st.selectbox("Color Space Model", ["RGB", "HSV", "YCbCr"])
    channel_idx = st.selectbox("Channel Index", ["Channel 1", "Channel 2", "Channel 3"])
    
    if st.button("Extract Color Channel"):
        idx = ["Channel 1", "Channel 2", "Channel 3"].index(channel_idx)
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        channels = img_ops.get_color_channels(np_proc, model)
        st.session_state.processed_image = img_ops.numpy_to_pil(channels[idx])
        st.success(f"Extracted {model} - {channel_idx}!")

    st.markdown("---")
    
    cmap_choice = st.selectbox("Select Pseudocolor Colormap", ["jet", "hot", "cool", "plasma", "inferno"])
    if st.button("Apply Pseudocolor Map"):
        apply_op(img_ops.apply_pseudocolor, cmap_choice)
        st.success(f"Mapped grayscale values to '{cmap_choice}' heatmap!")
        
    st.markdown("---")
    
    if st.button("Show RGB Histogram Plot"):
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        hist_img = img_ops.get_rgb_histogram(np_proc)
        st.session_state.processed_image = hist_img
        st.success("Generated RGB Histogram!")

# Category 6: Frequency Domain Filtering
with st.sidebar.expander("Frequency Domain (FFT)"):
    if st.button("Show FFT Magnitude Spectrum"):
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        f_shift = img_ops.compute_fft(np_proc)
        mag_spec = img_ops.get_fft_magnitude_spectrum(f_shift)
        st.session_state.processed_image = img_ops.numpy_to_pil(mag_spec).convert("RGB")
        st.success("Computed Fast Fourier Transform!")
        
    st.markdown("---")
    
    fft_filt_type = st.selectbox("FFT Filter Type", ["ideal_lpf", "ideal_hpf", "gaussian_lpf", "gaussian_hpf"])
    fft_cutoff = st.slider("FFT Cutoff Radius", 5, 150, 30)
    
    if st.button("Apply FFT Frequency Filter"):
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        reconstructed, _, _ = img_ops.apply_frequency_filter(np_proc, fft_filt_type, fft_cutoff)
        st.session_state.processed_image = img_ops.numpy_to_pil(reconstructed)
        st.success(f"Applied frequency filter: {fft_filt_type} (radius: {fft_cutoff})!")

# Category 7: Features & Segmentation
with st.sidebar.expander("Features & Segmentation"):
    if st.button("Detect Harris Corners"):
        apply_op(img_ops.harris_corner_detector)
        st.success("Harris Corners drawn in red!")
        
    if st.button("Compute LBP (Local Binary Patterns)"):
        apply_op(img_ops.compute_lbp)
        st.success("Generated texture LBP representation!")
        
    st.markdown("---")
    
    kmeans_k = st.slider("Number of Clusters K", 2, 8, 3)
    if st.button("Run K-Means Segmentation"):
        with st.spinner("Calculating K-Means centroids..."):
            apply_op(img_ops.kmeans_segmentation, kmeans_k)
        st.success(f"K-Means complete (K={kmeans_k})!")
        
    st.markdown("---")
    
    morph_op = st.selectbox("Morphological Operation", ["dilation", "erosion", "opening", "closing"])
    morph_thresh = st.slider("Morph Threshold (Binarization)", 0, 255, 128)
    if st.button("Apply Morphological Filter"):
        apply_op(img_ops.binary_morphology, morph_op, morph_thresh)
        st.success(f"Applied morphology: {morph_op}!")

# Category 8: Security & Compression
with st.sidebar.expander("Steganography & DCT Compression"):
    # LSB Steganography
    steg_msg = st.text_input("LSB Steganography Text", value="sarthak")
    steg_col1, steg_col2 = st.columns(2)
    with steg_col1:
        if st.button("Hide Message"):
            np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
            try:
                encoded = img_ops.lsb_steganography_encode(np_proc, steg_msg)
                st.session_state.processed_image = img_ops.numpy_to_pil(encoded)
                st.success("Encoded!")
            except ValueError as ve:
                st.error(str(ve))
    with steg_col2:
        if st.button("Extract Message"):
            np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
            msg = img_ops.lsb_steganography_decode(np_proc)
            if msg:
                st.info(f"Extracted Secret Message: {msg}")
            else:
                st.warning("No message signature found in Red LSB.")
                
    st.markdown("---")
    
    dct_quality = st.slider("DCT Compression Quality", 5, 100, 50)
    if st.button("Apply Block DCT Quantization"):
        np_proc = img_ops.pil_to_numpy(st.session_state.processed_image)
        compressed, mse, psnr = img_ops.block_dct_compression(np_proc, dct_quality)
        st.session_state.processed_image = img_ops.numpy_to_pil(compressed)
        st.success("Reconstructed block DCT image!")
        st.metric("Peak Signal-to-Noise Ratio (PSNR)", f"{psnr:.2f} dB")
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")

# ---------------------------------------------------------
# Main Panel Viewports
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(st.session_state.original_image, use_container_width=True)
    st.caption(f"Dimensions: {st.session_state.original_image.width}x{st.session_state.original_image.height}")

with col2:
    st.subheader("Processed Image")
    st.image(st.session_state.processed_image, use_container_width=True)
    
    # Download processed image as file (Standard browser download)
    buf = io.BytesIO()
    st.session_state.processed_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Download Processed Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png"
    )
