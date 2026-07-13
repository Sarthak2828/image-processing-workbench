import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

# ---------------------------------------------------------
# Conversions
# ---------------------------------------------------------
def pil_to_numpy(pil_img):
    return np.array(pil_img)

def numpy_to_pil(np_arr):
    # Ensure correct dtype and shape
    if np_arr.dtype != np.uint8:
        # Clip and convert
        np_arr = np.clip(np_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np_arr)

# ---------------------------------------------------------
# Point & Simple Operations
# ---------------------------------------------------------
def adjust_brightness(np_img, factor):
    """
    factor: float (0.0 to 2.0). 1.0 is original.
    """
    pil_img = numpy_to_pil(np_img)
    enhancer = ImageEnhance.Brightness(pil_img)
    enhanced = enhancer.enhance(factor)
    return pil_to_numpy(enhanced)

def adjust_contrast(np_img, factor):
    """
    factor: float (0.0 to 3.0). 1.0 is original.
    """
    pil_img = numpy_to_pil(np_img)
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced = enhancer.enhance(factor)
    return pil_to_numpy(enhanced)

def convert_to_grayscale(np_img):
    pil_img = numpy_to_pil(np_img)
    gray = ImageOps.grayscale(pil_img)
    # Return as 3-channel RGB to keep canvas display uniform
    return pil_to_numpy(gray.convert("RGB"))

def invert_image(np_img):
    pil_img = numpy_to_pil(np_img)
    inverted = ImageOps.invert(pil_img)
    return pil_to_numpy(inverted)

def apply_threshold(np_img, threshold_val):
    # Convert to grayscale first
    pil_img = numpy_to_pil(np_img)
    gray = ImageOps.grayscale(pil_img)
    gray_np = np.array(gray)
    binary = (gray_np >= threshold_val) * 255
    return pil_to_numpy(Image.fromarray(binary.astype(np.uint8)).convert("RGB"))

def otsu_threshold(np_img):
    """
    Computes Otsu's optimal threshold value for grayscale image.
    """
    pil_img = numpy_to_pil(np_img)
    gray = ImageOps.grayscale(pil_img)
    gray_np = np.array(gray)
    
    # Calculate histogram and probabilities
    hist, bin_edges = np.histogram(gray_np, bins=256, range=(0, 256))
    total = gray_np.size
    
    current_max = 0
    threshold = 0
    
    sum_total = np.sum(np.arange(256) * hist)
    sum_b = 0
    w_b = 0
    w_f = 0
    
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        
        # Between class variance
        var_between = w_b * w_f * (m_b - m_f) ** 2
        
        if var_between > current_max:
            current_max = var_between
            threshold = i
            
    # Apply threshold
    binary = (gray_np >= threshold) * 255
    return pil_to_numpy(Image.fromarray(binary.astype(np.uint8)).convert("RGB")), threshold

# ---------------------------------------------------------
# Neighborhood / Convolution Helper
# ---------------------------------------------------------
def convolve2d(img_2d, kernel):
    """
    Fast convolution using NumPy vectorized shift-sums.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img_2d, ((ph, ph), (pw, pw)), mode='edge')
    
    h, w = img_2d.shape
    out = np.zeros_like(img_2d, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i+h, j:j+w] * kernel[i, j]
    return out

def apply_kernel_rgb(np_img, kernel):
    # Handle both RGB and grayscale
    if len(np_img.shape) == 3:
        out = np.zeros_like(np_img, dtype=np.float32)
        for c in range(3):
            out[:, :, c] = convolve2d(np_img[:, :, c], kernel)
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        out = convolve2d(np_img, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# Spatial Filters
# ---------------------------------------------------------
def box_blur(np_img, size=5):
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return apply_kernel_rgb(np_img, kernel)

def gaussian_blur(np_img, size=5, sigma=1.5):
    # Generate gaussian kernel
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    kernel = kernel / np.sum(kernel)
    return apply_kernel_rgb(np_img, kernel)

def median_filter(np_img, size=3):
    if len(np_img.shape) == 3:
        out = np.zeros_like(np_img)
        for c in range(3):
            out[:, :, c] = median_filter_2d(np_img[:, :, c], size)
        return out
    else:
        return median_filter_2d(np_img, size)

def median_filter_2d(img_2d, size=3):
    h, w = img_2d.shape
    pad = size // 2
    padded = np.pad(img_2d, pad, mode='edge')
    shifts = []
    for i in range(size):
        for j in range(size):
            shifts.append(padded[i:i+h, j:j+w])
    return np.median(np.stack(shifts, axis=-1), axis=-1).astype(np.uint8)

def laplacian_sharpen(np_img, strength=1.0):
    # Laplacian kernel with center positive and surrounding negative
    # Kernel: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    # Let's parameterize the sharpening strength
    kernel = np.array([
        [0, -strength, 0],
        [-strength, 1 + 4*strength, -strength],
        [0, -strength, 0]
    ], dtype=np.float32)
    return apply_kernel_rgb(np_img, kernel)

def unsharp_mask(np_img, blur_size=5, strength=1.5):
    blurred = gaussian_blur(np_img, size=blur_size, sigma=blur_size/3.0)
    # Mask = Original - Blurred
    mask = np_img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = np_img.astype(np.float32) + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# Edges
# ---------------------------------------------------------
def edge_sobel(np_img):
    # Convert to grayscale first
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return pil_to_numpy(Image.fromarray(magnitude).convert("RGB"))

def edge_prewitt(np_img):
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    gx = convolve2d(gray, kx)
    gy = convolve2d(gray, ky)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return pil_to_numpy(Image.fromarray(magnitude).convert("RGB"))

def edge_log(np_img, size=9, sigma=1.4):
    """Laplacian of Gaussian edge detection"""
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    # Generate LoG kernel
    lim = size // 2
    y, x = np.ogrid[-lim:lim+1, -lim:lim+1]
    # LoG formula
    sigma_sq = sigma ** 2
    kernel = - ((x**2 + y**2 - 2 * sigma_sq) / (sigma_sq**2)) * np.exp(-(x**2 + y**2) / (2 * sigma_sq))
    # Normalize to 0-sum (high pass)
    kernel = kernel - np.mean(kernel)
    
    edges = convolve2d(gray, kernel)
    # Map edges to absolute scale
    edges = np.abs(edges)
    edges = (edges / np.max(edges)) * 255 if np.max(edges) > 0 else edges
    return pil_to_numpy(Image.fromarray(np.clip(edges, 0, 255).astype(np.uint8)).convert("RGB"))

def edge_dog(np_img, sigma1=1.0, sigma2=2.0):
    """Difference of Gaussians"""
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img))).astype(np.float32)
    
    # Apply two Gaussian blurs
    blur1 = gaussian_blur(gray, size=5, sigma=sigma1)
    blur2 = gaussian_blur(gray, size=9, sigma=sigma2)
    
    dog = blur1.astype(np.float32) - blur2.astype(np.float32)
    dog = np.abs(dog)
    # Normalize
    max_val = np.max(dog)
    if max_val > 0:
        dog = (dog / max_val) * 255
        
    return pil_to_numpy(Image.fromarray(np.clip(dog, 0, 255).astype(np.uint8)).convert("RGB"))

# ---------------------------------------------------------
# Transforms (Mirror, Rotate, Crop)
# ---------------------------------------------------------
def mirror_image(np_img, mode="horizontal"):
    pil_img = numpy_to_pil(np_img)
    if mode == "horizontal":
        flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    return pil_to_numpy(flipped)

def rotate_image(np_img, angle):
    pil_img = numpy_to_pil(np_img)
    rotated = pil_img.rotate(angle, expand=True)
    return pil_to_numpy(rotated)

def crop_image(np_img, x1, y1, x2, y2):
    h, w = np_img.shape[:2]
    # Keep inside bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        return np_img[y1:y2, x1:x2]
    return np_img

# ---------------------------------------------------------
# Color Model splits & Pseudocolor
# ---------------------------------------------------------
def get_color_channels(np_img, model="RGB"):
    pil_img = numpy_to_pil(np_img)
    if model == "RGB":
        channels = pil_img.split() # (R, G, B)
        return [pil_to_numpy(c.convert("RGB")) for c in channels]
    elif model == "HSV":
        hsv_img = pil_img.convert("HSV")
        channels = hsv_img.split() # (H, S, V)
        return [pil_to_numpy(c.convert("RGB")) for c in channels]
    elif model == "YCbCr":
        ycbcr_img = pil_img.convert("YCbCr")
        channels = ycbcr_img.split() # (Y, Cb, Cr)
        return [pil_to_numpy(c.convert("RGB")) for c in channels]
    return [np_img, np_img, np_img]

def apply_pseudocolor(np_img, colormap_name="jet"):
    # Convert to grayscale first
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    
    # Pure NumPy implementation of Jet colormap (no matplotlib needed)
    x = gray / 255.0
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    
    colored = np.stack([r, g, b], axis=-1)
    return (colored * 255).astype(np.uint8)

# ---------------------------------------------------------
# Frequency Domain (2D DFT)
# ---------------------------------------------------------
def compute_fft(np_img):
    # Convert to grayscale
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img))).astype(np.float32)
    # Compute 2D FFT and shift
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    return f_shift

def get_fft_magnitude_spectrum(f_shift):
    magnitude = np.abs(f_shift)
    # Log transform to compress range
    log_magnitude = np.log1p(magnitude)
    # Normalize to 0 - 255
    max_val = np.max(log_magnitude)
    if max_val > 0:
        log_magnitude = (log_magnitude / max_val) * 255
    return log_magnitude.astype(np.uint8)

def apply_frequency_filter(np_img, filter_type="ideal_lpf", cutoff=30):
    f_shift = compute_fft(np_img)
    h, w = f_shift.shape
    cy, cx = h // 2, w // 2
    
    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    d = np.sqrt((y - cy)**2 + (x - cx)**2)
    
    if filter_type == "ideal_lpf":
        mask = (d <= cutoff).astype(np.float32)
    elif filter_type == "ideal_hpf":
        mask = (d > cutoff).astype(np.float32)
    elif filter_type == "gaussian_lpf":
        mask = np.exp(- (d**2) / (2 * (cutoff**2)))
    elif filter_type == "gaussian_hpf":
        mask = 1 - np.exp(- (d**2) / (2 * (cutoff**2)))
    else:
        mask = np.ones((h, w), dtype=np.float32)
        
    filtered_shift = f_shift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(filtered_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    
    # Reconstructed image
    reconstructed = np.clip(img_back, 0, 255).astype(np.uint8)
    
    # We also return the magnitude spectrum of output, and the filter mask for display
    mask_display = (mask * 255).astype(np.uint8)
    # Render mask as RGB
    mask_rgb = pil_to_numpy(Image.fromarray(mask_display).convert("RGB"))
    
    reconstructed_rgb = pil_to_numpy(Image.fromarray(reconstructed).convert("RGB"))
    spectrum_rgb = pil_to_numpy(Image.fromarray(get_fft_magnitude_spectrum(filtered_shift)).convert("RGB"))
    
    return reconstructed_rgb, spectrum_rgb, mask_rgb

# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------
def harris_corner_detector(np_img, k=0.04, threshold_factor=0.01):
    # 1. Convert to grayscale
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img))).astype(np.float32)
    
    # 2. Image gradients
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Ix = convolve2d(gray, kx)
    Iy = convolve2d(gray, ky)
    
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # 3. Gaussian filter (weighted sum of neighbors)
    g_kernel = np.ones((5, 5), dtype=np.float32) / 25
    Sxx = convolve2d(Ix2, g_kernel)
    Syy = convolve2d(Iy2, g_kernel)
    Sxy = convolve2d(Ixy, g_kernel)
    
    # 4. Corner response R
    det = Sxx * Syy - Sxy**2
    trace = Sxx + Syy
    R = det - k * (trace ** 2)
    
    # 5. Thresholding and Local Maxima
    threshold = threshold_factor * np.max(R)
    corners = R > threshold
    
    # Draw detected corners on the original image
    pil_out = numpy_to_pil(np_img).copy()
    draw = ImageDraw.Draw(pil_out)
    
    # Locate indices
    y_indices, x_indices = np.where(corners)
    for y, x in zip(y_indices, x_indices):
        draw.ellipse([x-2, y-2, x+2, y+2], fill="red", outline="red")
        
    return pil_to_numpy(pil_out)

def compute_lbp(np_img):
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    h, w = gray.shape
    padded = np.pad(gray, 1, mode='edge')
    lbp = np.zeros_like(gray, dtype=np.uint8)
    
    shifts = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, 1),   (1, 1),  (1, 0),
        (1, -1),  (0, -1)
    ]
    
    for idx, (dy, dx) in enumerate(shifts):
        neighbor = padded[1+dy:1+dy+h, 1+dx:1+dx+w]
        lbp += ((neighbor >= gray) * (1 << idx)).astype(np.uint8)
        
    return pil_to_numpy(Image.fromarray(lbp).convert("RGB"))

# ---------------------------------------------------------
# Image Segmentation
# ---------------------------------------------------------
def kmeans_segmentation(np_img, k=3):
    h, w = np_img.shape[:2]
    if len(np_img.shape) == 2:
        pixels = np_img.reshape(-1, 1).astype(np.float32)
    else:
        pixels = np_img.reshape(-1, 3).astype(np.float32)
        
    n_pixels = pixels.shape[0]
    indices = np.random.choice(n_pixels, k, replace=False)
    centroids = pixels[indices]
    
    for iteration in range(10):
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([
            pixels[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
            for j in range(k)
        ])
        
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    segmented_pixels = centroids[labels].astype(np.uint8)
    segmented_img = segmented_pixels.reshape(np_img.shape)
    
    if len(segmented_img.shape) == 2:
        return pil_to_numpy(Image.fromarray(segmented_img).convert("RGB"))
    return segmented_img

def binary_morphology(np_img, op_type="dilation", threshold_val=128):
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img)))
    binary = (gray >= threshold_val).astype(np.uint8)
    h, w = binary.shape
    
    if op_type == "dilation":
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        out = np.zeros_like(binary)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                out = np.maximum(out, padded[1+dy:1+dy+h, 1+dx:1+dx+w])
    elif op_type == "erosion":
        padded = np.pad(binary, 1, mode='constant', constant_values=1)
        out = np.ones_like(binary)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                out = np.minimum(out, padded[1+dy:1+dy+h, 1+dx:1+dx+w])
    elif op_type == "opening":
        eroded = binary_morphology_helper(binary, "erosion")
        out = binary_morphology_helper(eroded, "dilation")
    elif op_type == "closing":
        dilated = binary_morphology_helper(binary, "dilation")
        out = binary_morphology_helper(dilated, "erosion")
    else:
        out = binary
        
    out_display = (out * 255).astype(np.uint8)
    return pil_to_numpy(Image.fromarray(out_display).convert("RGB"))

def binary_morphology_helper(binary, op_type):
    h, w = binary.shape
    if op_type == "dilation":
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        out = np.zeros_like(binary)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                out = np.maximum(out, padded[1+dy:1+dy+h, 1+dx:1+dx+w])
    else:
        padded = np.pad(binary, 1, mode='constant', constant_values=1)
        out = np.ones_like(binary)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                out = np.minimum(out, padded[1+dy:1+dy+h, 1+dx:1+dx+w])
    return out

# ---------------------------------------------------------
# Security & Compression
# ---------------------------------------------------------
def lsb_steganography_encode(np_img, secret_msg):
    output = np_img.copy()
    magic_prefix = "[STEG]"
    msg_to_encode = magic_prefix + secret_msg
    
    msg_bytes = (msg_to_encode + "\x00").encode('utf-8')
    bits = []
    for b in msg_bytes:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
            
    h, w = output.shape[:2]
    n_bits = len(bits)
    
    if n_bits > h * w:
        raise ValueError(f"Message too long! Image can hold max {h*w//8 - len(magic_prefix) - 1} chars.")
        
    flat_red = output[:, :, 0].flatten()
    for i in range(n_bits):
        flat_red[i] = (flat_red[i] & 0xFE) | bits[i]
        
    output[:, :, 0] = flat_red.reshape(h, w)
    return output

def lsb_steganography_decode(np_img):
    flat_red = np_img[:, :, 0].flatten()
    msg_bytes = bytearray()
    current_byte = 0
    bit_count = 0
    
    for pixel_val in flat_red:
        bit = int(pixel_val & 1)
        current_byte = (current_byte << 1) | bit
        bit_count += 1
        
        if bit_count == 8:
            if current_byte == 0:
                break
            msg_bytes.append(int(current_byte))
            current_byte = 0
            bit_count = 0
            if len(msg_bytes) > 2500:
                break
                
    try:
        decoded_str = msg_bytes.decode('utf-8', errors='ignore')
        magic_prefix = "[STEG]"
        if decoded_str.startswith(magic_prefix):
            return decoded_str[len(magic_prefix):]
        return ""
    except Exception:
        return ""

def compute_dct_matrix(N):
    C = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = np.sqrt(1.0 / N)
            else:
                C[i, j] = np.sqrt(2.0 / N) * np.cos(((2 * j + 1) * i * np.pi) / (2.0 * N))
    return C

def block_dct_compression(np_img, quality):
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img))).astype(np.float32)
    h, w = gray.shape
    h_block = (h // 8) * 8
    w_block = (w // 8) * 8
    gray_trunc = gray[:h_block, :w_block]
    
    C = compute_dct_matrix(8)
    C_T = C.T
    
    reconstructed = np.zeros_like(gray_trunc)
    cutoff = int((quality / 100.0) * 14)
    
    for y in range(0, h_block, 8):
        for x in range(0, w_block, 8):
            block = gray_trunc[y:y+8, x:x+8]
            dct_block = C @ block @ C_T
            for u in range(8):
                for v in range(8):
                    if u + v > cutoff:
                        dct_block[u, v] = 0.0
            idct_block = C_T @ dct_block @ C
            reconstructed[y:y+8, x:x+8] = idct_block
            
    mse = np.mean((gray_trunc - reconstructed) ** 2)
    if mse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
    reconstructed_clipped = np.clip(reconstructed, 0, 255).astype(np.uint8)
    reconstructed_rgb = pil_to_numpy(Image.fromarray(reconstructed_clipped).convert("RGB"))
    return reconstructed_rgb, mse, psnr

# ---------------------------------------------------------
# Histogram Equalization
# ---------------------------------------------------------
def equalize_histogram(np_img):
    pil_img = numpy_to_pil(np_img)
    hsv_img = pil_img.convert("HSV")
    h, s, v = hsv_img.split()
    v_np = np.array(v)
    hist, bins = np.histogram(v_np.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    v_equalized = np.interp(v_np.flatten(), bins[:-1], cdf_normalized).reshape(v_np.shape).astype(np.uint8)
    v_eq_pil = Image.fromarray(v_equalized)
    hsv_eq = Image.merge("HSV", (h, s, v_eq_pil))
    return pil_to_numpy(hsv_eq.convert("RGB"))

# ---------------------------------------------------------
# Canny Edge Detection (Pure NumPy Implementation)
# ---------------------------------------------------------
def edge_canny(np_img, low_threshold=20, high_threshold=50):
    gray = np.array(ImageOps.grayscale(numpy_to_pil(np_img))).astype(np.float32)
    g_kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=np.float32) / 273.0
    blurred = convolve2d(gray, g_kernel)
    
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    Ix = convolve2d(blurred, kx)
    Iy = convolve2d(blurred, ky)
    
    G = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix) * 180 / np.pi
    theta[theta < 0] += 180
    
    h, w = G.shape
    NMS = np.zeros_like(G)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = theta[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                n1 = G[i, j+1]
                n2 = G[i, j-1]
            elif (22.5 <= angle < 67.5):
                n1 = G[i+1, j-1]
                n2 = G[i-1, j+1]
            elif (67.5 <= angle < 112.5):
                n1 = G[i+1, j]
                n2 = G[i-1, j]
            else:
                n1 = G[i-1, j-1]
                n2 = G[i+1, j+1]
                
            if G[i, j] >= n1 and G[i, j] >= n2:
                NMS[i, j] = G[i, j]
                
    res = np.zeros_like(NMS, dtype=np.uint8)
    strong_i, strong_j = np.where(NMS >= high_threshold)
    weak_i, weak_j = np.where((NMS >= low_threshold) & (NMS < high_threshold))
    
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 50
    
    visited = np.zeros_like(res, dtype=bool)
    stack = list(zip(strong_i, strong_j))
    for r, c in stack:
        visited[r, c] = True
        
    while stack:
        r, c = stack.pop()
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if not visited[nr, nc] and res[nr, nc] == 50:
                        res[nr, nc] = 255
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                        
    res[res == 50] = 0
    return pil_to_numpy(Image.fromarray(res).convert("RGB"))

# ---------------------------------------------------------
# RGB Histogram Generator
# ---------------------------------------------------------
def get_rgb_histogram(np_img):
    # Calculate histograms for R, G, B using numpy
    h_r, _ = np.histogram(np_img[:, :, 0], bins=256, range=(0, 256))
    h_g, _ = np.histogram(np_img[:, :, 1], bins=256, range=(0, 256))
    h_b, _ = np.histogram(np_img[:, :, 2], bins=256, range=(0, 256))
    
    # Create image canvas matching the dark theme
    w, h = 600, 400
    img = Image.new("RGB", (w, h), "#2c3e50")
    draw = ImageDraw.Draw(img)
    
    # Draw a inner dark rectangle for the plot area
    draw.rectangle([50, 50, 550, 350], fill="#1a252f", outline="white", width=1)
    
    # Normalize histogram values to fit inside 280px height
    max_val = max(np.max(h_r), np.max(h_g), np.max(h_b), 1)
    
    # Draw horizontal grid lines
    for y_val in [125, 200, 275]:
        draw.line([50, y_val, 550, y_val], fill="#2c3e50", width=1)
        
    # Draw curves
    for hist, color in [(h_r, "red"), (h_g, "green"), (h_b, "blue")]:
        points = []
        for i in range(256):
            x = 50 + (i / 255.0) * 500
            y = 350 - (hist[i] / max_val) * 280
            points.append((x, y))
            
        for idx in range(len(points) - 1):
            draw.line([points[idx], points[idx+1]], fill=color, width=2)
            
    # Draw axes ticks
    for i in range(0, 257, 64):
        x = 50 + (i / 255.0) * 500
        draw.line([x, 350, x, 355], fill="white", width=1)
        
    # Draw Title
    draw.text((60, 20), "Real-Time RGB Histogram", fill="white")
    return img
