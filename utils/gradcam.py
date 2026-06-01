import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    """
    Gradient saliency map — works reliably with dual-branch models.
    Computes gradients of the predicted class score with respect
    to the input image, then aggregates spatially.
    """
    img_var = tf.Variable(tf.cast(img_array, tf.float32))

    with tf.GradientTape() as tape:
        preds    = model(img_var)
        pred_idx = int(tf.argmax(preds[0]))
        loss     = preds[:, pred_idx]

    grads = tape.gradient(loss, img_var)

    if grads is None:
        raise ValueError("Could not compute gradients.")

    # Average across colour channels to get a 2D spatial map
    heatmap = tf.reduce_mean(tf.abs(grads[0]), axis=-1).numpy()

    # Smooth for cleaner visualisation
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalise to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, int(pred_idx)


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