import base64
import io
import os
import sys
from fastapi import FastAPI, HTTPException, FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Add parent directory to path so we can import image_operations.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

@app.get("/")
async def get_index():
    # index.html is located in the parent directory
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_path = os.path.join(parent_dir, "index.html")
    return FileResponse(index_path)
