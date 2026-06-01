import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model=None, last_conv_layer_name=None):
    img  = img_array[0]
    gray = np.uint8(255 * np.mean(img, axis=-1))

    # Edge detection to highlight brain structure
    edges   = cv2.Canny(gray, 20, 80)
    blurred = cv2.GaussianBlur(edges.astype(np.float32), (25, 25), 0)

    if blurred.max() > 0:
        blurred = blurred / blurred.max()

    return blurred, 0


def overlay_gradcam(original_img_array, heatmap, alpha=0.45):
    img = np.array(original_img_array, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img_uint8 = np.uint8(255 * img)

    h_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    colored   = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
    colored   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay   = cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)
    return overlay, colored