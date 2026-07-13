import numpy as np
from PIL import Image
import image_operations as img_ops

def run_tests():
    print("Starting programmatic verification of all image processing functions...")
    
    # 1. Create a mock RGB test image (128x128)
    # A gradient with some geometric shapes drawn on it
    h, w = 128, 128
    y, x = np.ogrid[:h, :w]
    r = (x / w * 255).astype(np.uint8)
    g = (y / h * 255).astype(np.uint8)
    b = np.full((h, w), 128, dtype=np.uint8)
    img_np = np.stack([r, g, b], axis=-1)
    
    print("Mock image shape:", img_np.shape, "dtype:", img_np.dtype)
    
    # 2. Test basic point operations
    print("\n--- Testing Point Operations ---")
    
    res = img_ops.adjust_brightness(img_np, 1.5)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "adjust_brightness failed"
    print("[OK] adjust_brightness")
    
    res = img_ops.adjust_contrast(img_np, 1.2)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "adjust_contrast failed"
    print("[OK] adjust_contrast")
    
    res = img_ops.convert_to_grayscale(img_np)
    # Returns 3-channel RGB representation for unified UI canvas
    assert res.shape == img_np.shape and res.dtype == np.uint8, "convert_to_grayscale failed"
    print("[OK] convert_to_grayscale")
    
    res = img_ops.invert_image(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "invert_image failed"
    print("[OK] invert_image")
    
    res = img_ops.apply_threshold(img_np, 128)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "apply_threshold failed"
    print("[OK] apply_threshold")
    
    res, thresh = img_ops.otsu_threshold(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8 and 0 <= thresh <= 255, "otsu_threshold failed"
    print(f"[OK] otsu_threshold (computed threshold: {thresh})")
    
    res = img_ops.equalize_histogram(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "equalize_histogram failed"
    print("[OK] equalize_histogram")
    
    # 3. Test Spatial Filters
    print("\n--- Testing Spatial Filtering ---")
    
    res = img_ops.box_blur(img_np, size=3)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "box_blur failed"
    print("[OK] box_blur")
    
    res = img_ops.gaussian_blur(img_np, size=5, sigma=1.5)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "gaussian_blur failed"
    print("[OK] gaussian_blur")
    
    res = img_ops.median_filter(img_np, size=3)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "median_filter failed"
    print("[OK] median_filter")
    
    res = img_ops.laplacian_sharpen(img_np, strength=1.0)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "laplacian_sharpen failed"
    print("[OK] laplacian_sharpen")
    
    res = img_ops.unsharp_mask(img_np, blur_size=5, strength=1.5)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "unsharp_mask failed"
    print("[OK] unsharp_mask")
    
    # Edges
    res = img_ops.edge_sobel(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "edge_sobel failed"
    print("[OK] edge_sobel")
    
    res = img_ops.edge_prewitt(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "edge_prewitt failed"
    print("[OK] edge_prewitt")
    
    res = img_ops.edge_log(img_np, size=5, sigma=1.0)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "edge_log failed"
    print("[OK] edge_log")
    
    res = img_ops.edge_dog(img_np, sigma1=1.0, sigma2=2.0)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "edge_dog failed"
    print("[OK] edge_dog")
    
    res = img_ops.edge_canny(img_np, low_threshold=20, high_threshold=50)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "edge_canny failed"
    print("[OK] edge_canny")
    
    # 4. Test Transforms
    print("\n--- Testing Transforms ---")
    
    res = img_ops.mirror_image(img_np, "horizontal")
    assert res.shape == img_np.shape and res.dtype == np.uint8, "mirror_image horizontal failed"
    print("[OK] mirror_image (horizontal)")
    
    res = img_ops.mirror_image(img_np, "vertical")
    assert res.shape == img_np.shape and res.dtype == np.uint8, "mirror_image vertical failed"
    print("[OK] mirror_image (vertical)")
    
    res = img_ops.rotate_image(img_np, 90)
    # Rotating 90 degrees swap height/width
    assert res.shape == (img_np.shape[1], img_np.shape[0], img_np.shape[2]) and res.dtype == np.uint8, "rotate_image failed"
    print("[OK] rotate_image")
    
    res = img_ops.crop_image(img_np, 10, 20, 80, 90)
    assert res.shape == (70, 70, 3) and res.dtype == np.uint8, "crop_image failed"
    print("[OK] crop_image")
    
    # 5. Test Color & Pseudo-color
    print("\n--- Testing Color Processing ---")
    
    for model in ["RGB", "HSV", "YCbCr"]:
        channels = img_ops.get_color_channels(img_np, model)
        assert len(channels) == 3, f"get_color_channels {model} failed"
        for idx, c in enumerate(channels):
            assert c.shape == img_np.shape and c.dtype == np.uint8, f"channel {idx} in {model} has invalid shape/type"
        print(f"[OK] get_color_channels ({model})")
        
    res = img_ops.apply_pseudocolor(img_np, "jet")
    assert res.shape == img_np.shape and res.dtype == np.uint8, "apply_pseudocolor failed"
    print("[OK] apply_pseudocolor")
    
    # 6. Test Frequency Domain (FFT)
    print("\n--- Testing Frequency Domain ---")
    
    f_shift = img_ops.compute_fft(img_np)
    assert f_shift.shape == (h, w) and np.iscomplexobj(f_shift), "compute_fft failed"
    print("[OK] compute_fft")
    
    spectrum = img_ops.get_fft_magnitude_spectrum(f_shift)
    assert spectrum.shape == (h, w) and spectrum.dtype == np.uint8, "get_fft_magnitude_spectrum failed"
    print("[OK] get_fft_magnitude_spectrum")
    
    reconstructed, spectrum_filtered, mask_img = img_ops.apply_frequency_filter(img_np, "gaussian_lpf", cutoff=30)
    assert reconstructed.shape == img_np.shape and reconstructed.dtype == np.uint8, "apply_frequency_filter reconstructed failed"
    assert spectrum_filtered.shape == img_np.shape and spectrum_filtered.dtype == np.uint8, "apply_frequency_filter spectrum failed"
    assert mask_img.shape == img_np.shape and mask_img.dtype == np.uint8, "apply_frequency_filter mask failed"
    print("[OK] apply_frequency_filter (Gaussian LPF)")
    

    
    # 7. Test Feature Extraction
    print("\n--- Testing Feature Extraction ---")
    
    res = img_ops.harris_corner_detector(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "harris_corner_detector failed"
    print("[OK] harris_corner_detector")
    
    res = img_ops.compute_lbp(img_np)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "compute_lbp failed"
    print("[OK] compute_lbp")
    
    # 8. Test Segmentation
    print("\n--- Testing Segmentation ---")
    
    res = img_ops.kmeans_segmentation(img_np, k=3)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "kmeans_segmentation failed"
    print("[OK] kmeans_segmentation")
    

    
    res = img_ops.binary_morphology(img_np, "dilation", threshold_val=128)
    assert res.shape == img_np.shape and res.dtype == np.uint8, "binary_morphology failed"
    print("[OK] binary_morphology")
    
    # 9. Test Compression & Steganography
    print("\n--- Testing Steganography & Compression ---")
    
    test_msg = "Hello Steg 123!"
    encoded = img_ops.lsb_steganography_encode(img_np, test_msg)
    assert encoded.shape == img_np.shape and encoded.dtype == np.uint8, "lsb_steganography_encode failed"
    decoded = img_ops.lsb_steganography_decode(encoded)
    assert decoded == test_msg, f"lsb_steganography decode mismatch: expected '{test_msg}', got '{decoded}'"
    print("[OK] lsb_steganography (Encode & Decode)")
    
    reconstructed, mse, psnr = img_ops.block_dct_compression(img_np, quality=60)
    # DCT truncates dimensions to multiple of 8, so output shape will be matched to truncated size
    h_trunc = (h // 8) * 8
    w_trunc = (w // 8) * 8
    assert reconstructed.shape == (h_trunc, w_trunc, 3) and reconstructed.dtype == np.uint8, "block_dct_compression shape failed"
    assert mse >= 0 and psnr > 0, "block_dct_compression calculations failed"
    print(f"[OK] block_dct_compression (MSE: {mse:.2f}, PSNR: {psnr:.2f}dB)")
    
    hist_img = img_ops.get_rgb_histogram(img_np)
    assert isinstance(hist_img, Image.Image), "get_rgb_histogram failed to return PIL Image"
    print("[OK] get_rgb_histogram")
    
    print("\n=============================================")
    print("SUCCESS: All image processing operations passed!")
    print("=============================================")

if __name__ == "__main__":
    run_tests()
