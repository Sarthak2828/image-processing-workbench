import base64
import io
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Force python to search the api/ directory first, bypassing root image_operations.py (which contains matplotlib imports)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import backend operations
import image_operations as img_ops

app = FastAPI(title="UCS615 Image Processing Vercel API")

class ProcessRequest(BaseModel):
    image: str  # Base64 data URL
    operation: str
    params: dict = {}

def base64_to_pil(b64_str):
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def pil_to_base64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64_bytes = base64.b64encode(buf.getvalue())
    return "data:image/png;base64," + b64_bytes.decode('utf-8')

@app.post("/api/process")
@app.post("/process")
async def process_image(req: ProcessRequest):
    try:
        pil_img = base64_to_pil(req.image)
        np_img = img_ops.pil_to_numpy(pil_img)
        op = req.operation
        p = req.params
        
        status = "Operation completed."
        
        # 1. Point processing
        if op == "brightness_contrast":
            bright = float(p.get("brightness", 1.0))
            contrast = float(p.get("contrast", 1.0))
            res = img_ops.adjust_brightness(np_img, bright)
            res = img_ops.adjust_contrast(res, contrast)
            status = f"Adjusted parameters: Brightness={bright:.2f}, Contrast={contrast:.2f}"
        elif op == "grayscale":
            res = img_ops.convert_to_grayscale(np_img)
            status = "Converted workspace to grayscale."
        elif op == "negative":
            res = img_ops.invert_image(np_img)
            status = "Applied image negative inversion."
        elif op == "equalize":
            res = img_ops.equalize_histogram(np_img)
            status = "Applied color-preserving Histogram Equalization."
        elif op == "threshold":
            t_val = int(p.get("threshold", 128))
            res = img_ops.apply_threshold(np_img, t_val)
            status = f"Applied manual threshold (value: {t_val})."
        elif op == "otsu":
            res, t_val = img_ops.otsu_threshold(np_img)
            status = f"Otsu's auto threshold calculated: {t_val}. Applied binary threshold."
            
        # 2. Neighborhood filters
        elif op == "box_blur":
            size = int(p.get("size", 5))
            res = img_ops.box_blur(np_img, size)
            status = f"Applied Box Blur (size: {size}x{size})."
        elif op == "gaussian_blur":
            size = int(p.get("size", 5))
            res = img_ops.gaussian_blur(np_img, size, size / 3.0)
            status = f"Applied Gaussian Blur (size: {size}x{size}, sigma: {size/3.0:.2f})."
        elif op == "median":
            size = int(p.get("size", 5))
            res = img_ops.median_filter(np_img, size)
            status = f"Applied Median Filter (size: {size}x{size})."
        elif op == "sharpen":
            res = img_ops.laplacian_sharpen(np_img)
            status = "Applied Laplacian high-pass sharpening."
        elif op == "unsharp":
            res = img_ops.unsharp_mask(np_img)
            status = "Applied Unsharp Masking filter."
            
        # 3. Edge detection
        elif op == "sobel":
            res = img_ops.edge_sobel(np_img)
            status = "Applied Sobel edge detection."
        elif op == "prewitt":
            res = img_ops.edge_prewitt(np_img)
            status = "Applied Prewitt edge detection."
        elif op == "canny":
            res = img_ops.edge_canny(np_img, low_threshold=30, high_threshold=80)
            status = "Applied Canny edge detection (low=30, high=80)."
        elif op == "log":
            res = img_ops.edge_log(np_img)
            status = "Applied Laplacian of Gaussian (LoG) edge detection."
        elif op == "dog":
            res = img_ops.edge_dog(np_img)
            status = "Applied Difference of Gaussians (DoG) edge detection."
            
        # 4. Transforms
        elif op == "mirror":
            direction = p.get("direction", "horizontal")
            res = img_ops.mirror_image(np_img, direction)
            status = f"Mirrored image: {direction}."
        elif op == "rotate":
            angle = int(p.get("angle", 90))
            res = img_ops.rotate_image(np_img, angle)
            status = f"Rotated image by {-angle} degrees."
        elif op == "crop":
            left_pct = float(p.get("left_pct", 10)) / 100.0
            right_pct = float(p.get("right_pct", 90)) / 100.0
            w, h = pil_img.size
            x1 = int(w * left_pct)
            x2 = int(w * right_pct)
            y1 = int(h * left_pct)
            y2 = int(h * right_pct)
            res = img_ops.crop_image(np_img, x1, y1, x2, y2)
            status = f"Cropped image to bounds: ({x1}, {y1}) to ({x2}, {y2})."
            
        # 5. Color processing
        elif op == "extract":
            model = p.get("model", "RGB")
            idx = int(p.get("idx", 0))
            channels = img_ops.get_color_channels(np_img, model)
            res = channels[idx]
            status = f"Extracted {model} - Channel {idx+1}."
        elif op == "pseudocolor":
            cmap = p.get("cmap", "jet")
            res = img_ops.apply_pseudocolor(np_img, cmap)
            status = f"Mapped intensities to pseudocolor cmap: '{cmap}'."
        elif op == "rgb_hist":
            res_pil = img_ops.get_rgb_histogram(np_img)
            out_b64 = pil_to_base64(res_pil)
            return {"image": out_b64, "status": "Generated and rendered real-time RGB histogram."}
            
        # 6. Frequency Domain
        elif op == "fft":
            f_shift = img_ops.compute_fft(np_img)
            mag_spec = img_ops.get_fft_magnitude_spectrum(f_shift)
            res_pil = Image.fromarray(mag_spec).convert("RGB")
            out_b64 = pil_to_base64(res_pil)
            return {"image": out_b64, "status": "Displayed Fourier Transform log magnitude spectrum."}
        elif op == "freq_filter":
            filt_type = p.get("filt_type", "gaussian_lpf")
            cutoff = float(p.get("cutoff", 30))
            res, _, _ = img_ops.apply_frequency_filter(np_img, filt_type, cutoff)
            status = f"Applied frequency domain filter: {filt_type} (cutoff: {cutoff})."
            
        # 7. Features & Segmentation
        elif op == "harris":
            res = img_ops.harris_corner_detector(np_img)
            status = "Harris corner detection completed. Marked in red."
        elif op == "lbp":
            res = img_ops.compute_lbp(np_img)
            status = "Calculated Local Binary Pattern texture representation."
        elif op == "kmeans":
            k = int(p.get("k", 3))
            res = img_ops.kmeans_segmentation(np_img, k)
            status = f"K-Means segmentation complete with {k} clusters."
        elif op == "morphology":
            op_name = p.get("op", "dilation")
            t_val = int(p.get("threshold", 128))
            res = img_ops.binary_morphology(np_img, op_name, t_val)
            status = f"Applied morphology operation: {op_name} (binarized at threshold {t_val})."
            
        # 8. Steganography & DCT
        elif op == "steg_encode":
            msg = p.get("message", "")
            res = img_ops.lsb_steganography_encode(np_img, msg)
            status = f"Encoded secret message inside Red channel LSB."
        elif op == "steg_decode":
            msg = img_ops.lsb_steganography_decode(np_img)
            if msg:
                return {"image": req.image, "status": f"SUCCESS", "message": msg}
            else:
                raise HTTPException(status_code=400, detail="No hidden message signature found.")
        elif op == "dct":
            q = int(p.get("quality", 50))
            res, mse, psnr = img_ops.block_dct_compression(np_img, q)
            status = f"Block-DCT quantised (quality: {q}): MSE={mse:.2f}, PSNR={psnr:.2f}dB."
            
        else:
            raise HTTPException(status_code=400, detail="Unknown operation type.")

        # Convert result to PIL and then base64
        res_pil = img_ops.numpy_to_pil(res)
        out_b64 = pil_to_base64(res_pil)
        
        return {"image": out_b64, "status": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Embed frontend index.html inside python code as a string to bypass Vercel bundling limits completely
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UCS615: Digital Image Processing Workbench</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #1a252f;
            --sidebar-bg: #2c3e50;
            --panel-bg: #2c3e50;
            --text-color: #ecf0f1;
            --accent-blue: #3498db;
            --accent-blue-hover: #2980b9;
            --accent-green: #2ecc71;
            --accent-green-hover: #27ae60;
            --accent-red: #e74c3c;
            --accent-red-hover: #c0392b;
            --accent-orange: #f39c12;
            --accent-orange-hover: #d35400;
            --border-color: #34495e;
            --status-bg: #34495e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color) var(--sidebar-bg);
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            height: 100vh;
            display: grid;
            grid-template-rows: 1fr 28px;
            overflow: hidden;
        }

        #loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(26, 37, 47, 0.7);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
        }

        #loading-overlay.active {
            opacity: 1;
            pointer-events: auto;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            border-top-color: var(--accent-blue);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        #loading-overlay h2 {
            margin-top: 20px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: white;
        }

        .workspace {
            display: grid;
            grid-template-columns: 360px 1fr;
            height: 100%;
            overflow: hidden;
        }

        .sidebar {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: grid;
            grid-template-rows: auto 1fr;
            overflow: hidden;
            height: 100%;
        }

        .sidebar-fixed {
            padding: 15px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 10px;
        }

        .btn-upload-label {
            flex: 1;
            background-color: var(--accent-green);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.2s;
            display: block;
        }

        .btn-upload-label:hover {
            background-color: var(--accent-green-hover);
        }

        #file-input {
            display: none;
        }

        .btn-reset {
            flex: 1;
            background-color: var(--accent-red);
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: background-color 0.2s;
        }

        .btn-reset:hover {
            background-color: var(--accent-red-hover);
        }

        .sidebar-scroll {
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .group-label {
            font-family: 'Outfit', sans-serif;
            font-size: 13px;
            font-weight: 800;
            color: var(--accent-blue);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .sub-label {
            font-size: 11px;
            font-weight: 600;
            color: #95a5a6;
            margin: 6px 0 4px 0;
        }

        .ctrl-group {
            background-color: rgba(0, 0, 0, 0.15);
            border-radius: 8px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .slider-container {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #bdc3c7;
        }

        input[type="range"] {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: #4f637a;
            border-radius: 3px;
            outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--accent-blue);
            cursor: pointer;
            transition: background 0.2s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            background: #5dade2;
        }

        .btn-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .btn-grid-3 {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 6px;
        }

        .btn-action {
            background-color: rgba(255, 255, 255, 0.08);
            color: var(--text-color);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
        }

        .btn-action:hover {
            background-color: var(--accent-blue);
            border-color: var(--accent-blue);
        }

        .btn-action-full {
            grid-column: span 2;
        }

        .btn-orange {
            background-color: var(--accent-orange);
            border-color: var(--accent-orange);
        }
        .btn-orange:hover {
            background-color: var(--accent-orange-hover);
            border-color: var(--accent-orange-hover);
        }

        select {
            background-color: #34495e;
            color: white;
            border: 1px solid var(--border-color);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
            width: 100%;
            cursor: pointer;
        }

        .steg-input {
            background-color: #34495e;
            color: white;
            border: 1px solid var(--border-color);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 12px;
            outline: none;
            width: 100%;
        }

        .viewports {
            padding: 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            height: 100%;
            overflow: hidden;
        }

        .panel-card {
            background-color: var(--panel-bg);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: grid;
            grid-template-rows: auto 1fr;
            padding: 15px;
            overflow: hidden;
            height: 100%;
        }

        .panel-title {
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 600;
            text-align: center;
            margin-bottom: 10px;
        }

        .canvas-container {
            background-color: #2c3e50;
            border-radius: 8px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            position: relative;
            height: 100%;
            border: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .canvas-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }

        .status-bar {
            background-color: var(--status-bg);
            border-top: 1px solid var(--border-color);
            padding: 4px 15px;
            font-size: 13px;
            color: #bdc3c7;
            display: flex;
            align-items: center;
            height: 28px;
        }
    </style>
</head>
<body>

    <div id="loading-overlay">
        <div class="spinner"></div>
        <h2>Processing Image...</h2>
    </div>

    <div class="workspace">
        <div class="sidebar">
            <div class="sidebar-fixed">
                <label for="file-input" class="btn-upload-label">Upload Image</label>
                <input type="file" id="file-input" accept="image/*">
                <button class="btn-reset" onclick="web_reset_image()">Reset Image</button>
            </div>
            
            <div class="sidebar-scroll">
                <div class="ctrl-group">
                    <div class="group-label">Spatial Point Processing</div>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Brightness</span>
                            <span id="val-bright">1.0</span>
                        </div>
                        <input type="range" id="slider-bright" min="0.0" max="2.0" step="0.05" value="1.0" oninput="document.getElementById('val-bright').innerText = this.value; web_on_point_change()">
                    </div>

                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Contrast</span>
                            <span id="val-contrast">1.0</span>
                        </div>
                        <input type="range" id="slider-contrast" min="0.0" max="3.0" step="0.05" value="1.0" oninput="document.getElementById('val-contrast').innerText = this.value; web_on_point_change()">
                    </div>

                    <div class="btn-grid">
                        <button class="btn-action" onclick="applyOperation('grayscale')">Grayscale (B&W)</button>
                        <button class="btn-action" onclick="applyOperation('negative')">Negative (Invert)</button>
                        <button class="btn-action btn-action-full" onclick="applyOperation('equalize')">Equalize Histogram</button>
                    </div>

                    <div class="sub-label">Binary Thresholding</div>
                    <div style="display: flex; gap: 8px; align-items: center;">
                        <input type="range" id="slider-thresh" min="0" max="255" value="128" style="flex: 1;" oninput="document.getElementById('val-thresh').innerText = this.value">
                        <span id="val-thresh" style="font-size: 12px; min-width: 25px;">128</span>
                        <button class="btn-action" style="padding: 6px 10px;" onclick="applyThreshold()">Apply</button>
                    </div>
                    <button class="btn-action" onclick="applyOperation('otsu')">Otsu Auto Thresholding</button>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Spatial Filters (Neighborhood)</div>
                    <div class="sub-label">Image Smoothing (Blur)</div>
                    <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 5px;">
                        <span style="font-size: 12px; color: #bdc3c7;">Kernel Size:</span>
                        <select id="select-blur-size" style="width: auto;">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="7">7</option>
                            <option value="9">9</option>
                            <option value="15">15</option>
                        </select>
                    </div>
                    <div class="btn-grid-3">
                        <button class="btn-action" onclick="applyBlur('box_blur')">Box</button>
                        <button class="btn-action" onclick="applyBlur('gaussian_blur')">Gaussian</button>
                        <button class="btn-action" onclick="applyBlur('median')">Median</button>
                    </div>

                    <div class="sub-label">Image Sharpening</div>
                    <div class="btn-grid">
                        <button class="btn-action" onclick="applyOperation('sharpen')">Laplacian</button>
                        <button class="btn-action" onclick="applyOperation('unsharp')">Unsharp Mask</button>
                    </div>

                    <div class="sub-label">Edge Detection</div>
                    <div class="btn-grid-3">
                        <button class="btn-action" onclick="applyOperation('sobel')">Sobel</button>
                        <button class="btn-action" onclick="applyOperation('prewitt')">Prewitt</button>
                        <button class="btn-action" onclick="applyOperation('canny')">Canny</button>
                    </div>
                    <div class="btn-grid">
                        <button class="btn-action" onclick="applyOperation('log')">LoG</button>
                        <button class="btn-action" onclick="applyOperation('dog')">DoG</button>
                    </div>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Transforms & Mirroring</div>
                    <div class="btn-grid">
                        <button class="btn-action" onclick="applyOperation('mirror', {direction: 'horizontal'})">Mirror H</button>
                        <button class="btn-action" onclick="applyOperation('mirror', {direction: 'vertical'})">Mirror V</button>
                        <button class="btn-action" onclick="applyOperation('rotate', {angle: 90})">Rotate 90 L</button>
                        <button class="btn-action" onclick="applyOperation('rotate', {angle: -90})">Rotate 90 R</button>
                    </div>
                    
                    <div class="sub-label">Simple Web Bounding Box Crop</div>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Crop Left Bounds %</span>
                            <span id="val-crop-left">10</span>
                        </div>
                        <input type="range" id="slider-crop-left" min="0" max="49" value="10" oninput="document.getElementById('val-crop-left').innerText = this.value">
                    </div>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Crop Right Bounds %</span>
                            <span id="val-crop-right">90</span>
                        </div>
                        <input type="range" id="slider-crop-right" min="50" max="100" value="90" oninput="document.getElementById('val-crop-right').innerText = this.value">
                    </div>
                    <button class="btn-action btn-orange" onclick="applyCrop()">Apply Crop Bounds</button>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Color Processing</div>
                    <div class="sub-label">Extract Color Channel</div>
                    <div style="display: flex; gap: 8px; margin-bottom: 5px;">
                        <select id="select-color-model">
                            <option value="RGB">RGB</option>
                            <option value="HSV">HSV</option>
                            <option value="YCbCr">YCbCr</option>
                        </select>
                        <select id="select-channel-idx">
                            <option value="0">Channel 1</option>
                            <option value="1">Channel 2</option>
                            <option value="2">Channel 3</option>
                        </select>
                    </div>
                    <button class="btn-action" onclick="applyChannelExtract()">Extract Channel</button>

                    <div class="sub-label">Pseudo-color Mapping</div>
                    <div style="display: flex; gap: 8px;">
                        <select id="select-cmap" style="flex: 1;">
                            <option value="jet">jet</option>
                            <option value="hot">hot</option>
                            <option value="cool">cool</option>
                            <option value="plasma">plasma</option>
                            <option value="inferno">inferno</option>
                        </select>
                        <button class="btn-action" onclick="applyPseudocolor()">Apply</button>
                    </div>
                    
                    <button class="btn-action btn-action-full" style="margin-top: 5px;" onclick="applyOperation('rgb_hist')">Show RGB Histogram</button>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Frequency Domain (DFT)</div>
                    <button class="btn-action" onclick="applyOperation('fft')">Show Log FFT Spectrum</button>
                    
                    <div class="sub-label">Fourier Domain Filters</div>
                    <select id="select-freq-filt" style="margin-bottom: 5px;">
                        <option value="ideal_lpf">ideal_lpf</option>
                        <option value="ideal_hpf">ideal_hpf</option>
                        <option value="gaussian_lpf" selected>gaussian_lpf</option>
                        <option value="gaussian_hpf">gaussian_hpf</option>
                    </select>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Cutoff Radius</span>
                            <span id="val-cutoff">30</span>
                        </div>
                        <input type="range" id="slider-cutoff" min="5" max="150" value="30" oninput="document.getElementById('val-cutoff').innerText = this.value">
                    </div>
                    <button class="btn-action" style="margin-top: 5px;" onclick="applyFrequencyFilter()">Apply FFT Filter</button>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Features & Segmentation</div>
                    <div class="btn-grid">
                        <button class="btn-action" onclick="applyOperation('harris')">Harris Corners</button>
                        <button class="btn-action" onclick="applyOperation('lbp')">LBP Texture</button>
                    </div>

                    <div class="sub-label">K-Means Clustering Segmentation</div>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Clusters K</span>
                            <span id="val-kmeans-k">3</span>
                        </div>
                        <input type="range" id="slider-kmeans-k" min="2" max="8" value="3" oninput="document.getElementById('val-kmeans-k').innerText = this.value">
                    </div>
                    <button class="btn-action" style="margin-top: 5px;" onclick="applyKMeans()">Run K-Means</button>

                    <div class="sub-label">Binary Morphology</div>
                    <div style="display: flex; gap: 8px;">
                        <select id="select-morph-op" style="flex: 1;">
                            <option value="dilation">dilation</option>
                            <option value="erosion">erosion</option>
                            <option value="opening">opening</option>
                            <option value="closing">closing</option>
                        </select>
                        <button class="btn-action" onclick="applyMorphology()">Apply</button>
                    </div>
                </div>

                <div class="ctrl-group">
                    <div class="group-label">Steganography & Compression</div>
                    <div class="sub-label">LSB Steganography</div>
                    <input type="text" id="txt-steg-msg" class="steg-input" placeholder="Secret message to hide...">
                    <div class="btn-grid" style="margin-top: 5px;">
                        <button class="btn-action" onclick="applyStegEncode()">Encode</button>
                        <button class="btn-action" onclick="applyStegDecode()">Decode</button>
                    </div>

                    <div class="sub-label">Block DCT Compression</div>
                    <div class="slider-container">
                        <div class="slider-header">
                            <span>Quality</span>
                            <span id="val-dct-quality">50</span>
                        </div>
                        <input type="range" id="slider-dct-quality" min="5" max="100" value="50" oninput="document.getElementById('val-dct-quality').innerText = this.value">
                    </div>
                    <button class="btn-action" style="margin-top: 5px;" onclick="applyDCT()">Apply Block DCT</button>
                </div>
            </div>
        </div>

        <div class="viewports">
            <div class="panel-card">
                <div class="panel-title">Original Image</div>
                <div class="canvas-container">
                    <img id="img-orig" alt="No image loaded.">
                </div>
            </div>

            <div class="panel-card">
                <div class="panel-title">Processed Image</div>
                <div class="canvas-container">
                    <img id="img-proc" alt="No image loaded.">
                </div>
            </div>
        </div>
    </div>

    <div class="status-bar" id="status-text">
        Status: Ready. Load an image or modify parameters.
    </div>

    <script>
        let originalBase64 = null;
        let currentBase64 = null;
        const loader = document.getElementById('loading-overlay');

        function set_status(text) {
            document.getElementById("status-text").innerText = "Status: " + text;
        }

        document.getElementById('file-input').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(evt) {
                    originalBase64 = evt.target.result;
                    currentBase64 = originalBase64;
                    document.getElementById('img-orig').src = originalBase64;
                    document.getElementById('img-proc').src = originalBase64;
                    document.getElementById("slider-bright").value = "1.0";
                    document.getElementById("val-bright").innerText = "1.0";
                    document.getElementById("slider-contrast").value = "1.0";
                    document.getElementById("val-contrast").innerText = "1.0";
                    set_status("Loaded image successfully.");
                };
                reader.readAsDataURL(file);
            }
        });

        function web_reset_image() {
            if (originalBase64) {
                currentBase64 = originalBase64;
                document.getElementById('img-proc').src = originalBase64;
                document.getElementById("slider-bright").value = "1.0";
                document.getElementById("val-bright").innerText = "1.0";
                document.getElementById("slider-contrast").value = "1.0";
                document.getElementById("val-contrast").innerText = "1.0";
                set_status("Reverted to original image.");
            }
        }

        async function applyOperation(operation, params = {}) {
            if (!currentBase64) {
                alert("Please upload an image first.");
                return;
            }

            loader.classList.add('active');
            set_status("Processing... Please wait.");
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: currentBase64,
                        operation: operation,
                        params: params
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    currentBase64 = data.image;
                    document.getElementById('img-proc').src = currentBase64;
                    set_status(data.status);
                } else {
                    alert("Error: " + data.detail);
                    set_status("Operation failed.");
                }
            } catch (err) {
                alert("Server communication error: " + err.message);
                set_status("Connection error.");
            } finally {
                loader.classList.remove('active');
            }
        }

        function web_on_point_change() {
            if (!originalBase64) return;
            const bVal = parseFloat(document.getElementById("slider-bright").value);
            const cVal = parseFloat(document.getElementById("slider-contrast").value);
            applyOperation('brightness_contrast', { brightness: bVal, contrast: cVal });
        }

        function applyThreshold() {
            const tVal = parseInt(document.getElementById("slider-thresh").value);
            applyOperation('threshold', { threshold: tVal });
        }

        function applyBlur(type) {
            const size = parseInt(document.getElementById("select-blur-size").value);
            applyOperation(type, { size: size });
        }

        function applyCrop() {
            const left = parseFloat(document.getElementById("slider-crop-left").value);
            const right = parseFloat(document.getElementById("slider-crop-right").value);
            applyOperation('crop', { left_pct: left, right_pct: right });
        }

        function applyChannelExtract() {
            const model = document.getElementById("select-color-model").value;
            const idx = parseInt(document.getElementById("select-channel-idx").value);
            applyOperation('extract', { model: model, idx: idx });
        }

        function applyPseudocolor() {
            const cmap = document.getElementById("select-cmap").value;
            applyOperation('pseudocolor', { cmap: cmap });
        }

        function applyFrequencyFilter() {
            const filtType = document.getElementById("select-freq-filt").value;
            const cutoff = parseFloat(document.getElementById("slider-cutoff").value);
            applyOperation('freq_filter', { filt_type: filtType, cutoff: cutoff });
        }

        function applyKMeans() {
            const k = parseInt(document.getElementById("slider-kmeans-k").value);
            applyOperation('kmeans', { k: k });
        }

        function applyMorphology() {
            const op = document.getElementById("select-morph-op").value;
            const threshold = parseInt(document.getElementById("slider-thresh").value);
            applyOperation('morphology', { op: op, threshold: threshold });
        }

        function applyStegEncode() {
            const msg = document.getElementById("txt-steg-msg").value;
            if (!msg) {
                alert("Please enter a secret message to hide.");
                return;
            }
            applyOperation('steg_encode', { message: msg });
        }

        async function applyStegDecode() {
            if (!currentBase64) {
                alert("Please upload an image first.");
                return;
            }
            
            loader.classList.add('active');
            set_status("Decoding steganography... Please wait.");
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: currentBase64,
                        operation: 'steg_decode',
                        params: {}
                    })
                });

                const data = await response.json();
                if (response.ok) {
                    alert("Extracted Secret Message: " + data.message);
                    set_status("Decoded message successfully: " + data.message);
                } else {
                    alert("No message signature found in Red LSB.");
                    set_status("No hidden message found.");
                }
            } catch (err) {
                alert("Server communication error: " + err.message);
            } finally {
                loader.classList.remove('active');
            }
        }

        function applyDCT() {
            const q = parseInt(document.getElementById("slider-dct-quality").value);
            applyOperation('dct', { quality: q });
        }

        function generateDefaultImage() {
            const canvas = document.createElement('canvas');
            canvas.width = 450;
            canvas.height = 450;
            const ctx = canvas.getContext('2d');
            for (let i = 0; i < 450; i++) {
                const r = Math.floor((i / 450.0) * 200);
                const g = Math.floor(((450 - i) / 450.0) * 200);
                const b = 150;
                ctx.strokeStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.beginPath();
                ctx.moveTo(0, i);
                ctx.lineTo(450, i);
                ctx.stroke();
            }
            ctx.fillStyle = 'rgb(255, 215, 0)';
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.arc(225, 225, 75, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'rgb(30, 144, 255)';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.rect(50, 50, 70, 70);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = 'white';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.rect(320, 50, 80, 70);
            ctx.fill();
            ctx.stroke();
            const base64 = canvas.toDataURL('image/png');
            originalBase64 = base64;
            currentBase64 = base64;
            document.getElementById('img-orig').src = base64;
            document.getElementById('img-proc').src = base64;
        }
        generateDefaultImage();
    </script>
</body>
</html>
"""

@app.get("/")
async def get_index():
    return HTMLResponse(content=HTML_CONTENT)
