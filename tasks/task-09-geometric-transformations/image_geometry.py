# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:

    if img.ndim != 2:
        raise ValueError("A imagem deve ser um array 2D (grayscale).")
    img = np.asarray(img)
    h, w = img.shape
    dtype = img.dtype

     # ---------- 1) Translation----------
    dy, dx = 20, 30 
    translated = np.zeros_like(img)
    ys = slice(dy, h)
    xs = slice(dx, w)
    translated[ys, xs] = img[:h - dy, :w - dx]

    # ---------- 2) Rotation 90° clockwise ----------
    rotated = np.rot90(img, k=-1)

    # ---------- 3) Horizontal stretch ----------
    scale = 1.5
    new_w = max(1, int(round(w * scale)))
    x_src = np.minimum((np.round(np.arange(new_w) / scale)).astype(int), w - 1)
    stretched = img[:, x_src]

    # ---------- 4) Horizontal mirror (flip LR) ----------
    mirrored = img[:, ::-1]

    # ---------- 5) Barrel distortion (radial) ----------
    k = -0.3
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    rnorm = max(cx, cy) if max(cx, cy) > 0 else 1.0
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    uo = (xx - cx) / rnorm
    vo = (yy - cy) / rnorm
    r2 = uo**2 + vo**2

    factor = 1.0 + k * r2
    ui = uo / factor
    vi = vo / factor

    xi = ui * rnorm + cx
    yi = vi * rnorm + cy

    xi_nn = np.rint(xi).astype(int)
    yi_nn = np.rint(yi).astype(int)

    distorted = img[np.clip(yi_nn, 0, h - 1), np.clip(xi_nn, 0, w - 1)]

    return {
        "translated": translated.astype(dtype, copy=False),
        "rotated": rotated.astype(dtype, copy=False),
        "stretched": stretched.astype(dtype, copy=False),
        "mirrored": mirrored.astype(dtype, copy=False),
        "distorted": distorted.astype(dtype, copy=False),
    }