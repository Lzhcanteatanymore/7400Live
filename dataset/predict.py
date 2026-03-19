import os
import sys
import cv2
import joblib
import mahotas
import numpy as np
import pandas as pd

from pyefd import elliptic_fourier_descriptors
from skimage import morphology, measure
from skimage.filters import threshold_otsu

TARGET_SIZE = (90, 90)
MODEL_PATH = "./results/best_model.pkl"


def load_and_normalize(path, target_size=TARGET_SIZE):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    img_resized = cv2.GaussianBlur(img_resized, (3, 3), 0)

    img_float = img_resized.astype(np.float32)
    mean_rgb = img_float.reshape(-1, 3).mean(axis=0)
    mean_gray = mean_rgb.mean()
    scale = mean_gray / (mean_rgb + 1e-6)
    img_wb = img_float * scale
    img_wb = np.clip(img_wb, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_wb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v2 = clahe.apply(v)
    hsv2 = cv2.merge((h, s, v2))
    img_norm = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    return img_norm.astype(np.uint8)


def _corner_samples(arr: np.ndarray, patch: int = 10) -> np.ndarray:
    H, W = arr.shape[:2]
    p = int(min(patch, H // 2, W // 2))
    if p <= 0:
        return arr.reshape(-1, arr.shape[-1])

    corners = np.concatenate(
        [
            arr[:p, :p].reshape(-1, arr.shape[-1]),
            arr[:p, -p:].reshape(-1, arr.shape[-1]),
            arr[-p:, :p].reshape(-1, arr.shape[-1]),
            arr[-p:, -p:].reshape(-1, arr.shape[-1]),
        ],
        axis=0,
    )
    return corners


def segment_fruit(
    img_rgb: np.ndarray,
    patch: int = 10,
    bg_quantile: float = 0.995,
    v_min_floor: int = 20,
    reg: float = 1e-3,
) -> np.ndarray:
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    bg_hsv = _corner_samples(hsv, patch=patch).astype(np.float32)
    bg_lab = _corner_samples(lab, patch=patch).astype(np.float32)

    bg_ab = bg_lab[:, 1:3]
    mu = bg_ab.mean(axis=0)

    cov = np.cov(bg_ab.T)
    if cov.shape != (2, 2):
        cov = np.eye(2, dtype=np.float32)
    cov = cov.astype(np.float32) + float(reg) * np.eye(2, dtype=np.float32)

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    ab = lab[..., 1:3].reshape(-1, 2).astype(np.float32)
    diff = ab - mu.reshape(1, 2)
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff).reshape(H, W)

    bg_v_mean = float(bg_hsv[:, 2].mean())
    bg_v_std = float(bg_hsv[:, 2].std())
    v_min = int(max(v_min_floor, bg_v_mean - 2.0 * bg_v_std))
    V = hsv[..., 2].astype(np.int32)

    bg_d2_samp = _corner_samples(d2[..., None], patch=patch).reshape(-1)
    tau = float(np.quantile(bg_d2_samp, bg_quantile))

    raw = ((d2 > tau) & (V >= v_min)).astype(np.uint8) * 255

    frac = raw.mean() / 255.0
    if frac < 0.03 or frac > 0.90:
        try:
            tau2 = float(threshold_otsu(d2.astype(np.float32)))
            raw2 = ((d2 > tau2) & (V >= v_min)).astype(np.uint8) * 255
            frac2 = raw2.mean() / 255.0
            if 0.03 <= frac2 <= 0.90:
                raw = raw2
                frac = frac2
        except Exception:
            pass

    if frac < 0.03:
        S = hsv[..., 1].astype(np.int32)
        bg_s_mean = float(bg_hsv[:, 1].mean())
        bg_s_std = float(bg_hsv[:, 1].std())
        s_thresh = int(np.clip(bg_s_mean + 2.5 * bg_s_std, 25, 240))
        raw = ((S >= s_thresh) & (V >= v_min)).astype(np.uint8) * 255

    return raw


def refine_mask(
    mask: np.ndarray,
    hole_area: int = 200,
    min_object_size: int = 300,
    closing_radius: int = 5,
    opening_radius: int = 2,
) -> np.ndarray:
    mask_bool = mask > 0
    mask_filled = morphology.remove_small_holes(mask_bool, area_threshold=hole_area)
    mask_clean = morphology.remove_small_objects(mask_filled, min_size=min_object_size)

    if closing_radius > 0:
        mask_clean = morphology.closing(mask_clean, morphology.disk(closing_radius))
    if opening_radius > 0:
        mask_clean = morphology.opening(mask_clean, morphology.disk(opening_radius))

    labeled = measure.label(mask_clean)
    if labeled.max() == 0:
        return mask_filled.astype(np.uint8) * 255

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    mask_final = labeled == largest.label
    return mask_final.astype(np.uint8) * 255


def compute_shape_features(mask_clean):
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contour found in mask.")

    cnt = max(contours, key=cv2.contourArea)
    pts = cnt[:, 0, :].astype(np.float32)

    ksize = 5
    pts_smooth = np.copy(pts)
    pts_smooth[:, 0] = cv2.GaussianBlur(pts[:, 0], (ksize, 1), 0).flatten()
    pts_smooth[:, 1] = cv2.GaussianBlur(pts[:, 1], (ksize, 1), 0).flatten()

    cnt_smooth = pts_smooth.reshape(-1, 1, 2).astype(np.int32)

    area = cv2.contourArea(cnt_smooth)
    perimeter = cv2.arcLength(cnt_smooth, True)
    perimeter_norm = perimeter / (np.sqrt(area) + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

    hull = cv2.convexHull(cnt_smooth)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    if len(cnt_smooth) >= 5:
        ellipse = cv2.fitEllipse(cnt_smooth)
        (_, axes, _) = ellipse
        major_axis = max(axes)
        minor_axis = min(axes)
        eccentricity = np.sqrt(1 - (minor_axis / (major_axis + 1e-6))**2)
    else:
        eccentricity = 0.0

    dx = np.gradient(pts_smooth[:, 0])
    dy = np.gradient(pts_smooth[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.mean(
        np.abs(dx * ddy - dy * ddx) /
        (dx * dx + dy * dy + 1e-6) ** 1.5
    )

    efd = elliptic_fourier_descriptors(pts_smooth, order=10, normalize=True).flatten()

    moments = cv2.moments(cnt_smooth)
    cx = moments["m10"] / (moments["m00"] + 1e-6)
    cy = moments["m01"] / (moments["m00"] + 1e-6)

    radial = np.sqrt((pts_smooth[:, 0] - cx) ** 2 + (pts_smooth[:, 1] - cy) ** 2)
    radial_norm = radial / (np.mean(radial) + 1e-6)
    radial_fft = np.abs(np.fft.fft(radial_norm))
    radial_descriptor = radial_fft[:15]

    radius = min(mask_clean.shape) // 2
    zernike = mahotas.features.zernike_moments(mask_clean, radius, degree=8)

    return {
        "area": area,
        "perimeter": perimeter,
        "perimeter_norm": perimeter_norm,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "curvature": curvature,
        "efd": efd,
        "radial_signature": radial_descriptor,
        "zernike": zernike,
    }


def circular_mean_std(h):
    theta = (h.astype(np.float32) / 180.0) * 2.0 * np.pi
    sin_vals = np.sin(theta)
    cos_vals = np.cos(theta)

    sin_m = np.mean(sin_vals)
    cos_m = np.mean(cos_vals)

    mean_theta = np.arctan2(sin_m, cos_m)
    if mean_theta < 0:
        mean_theta += 2.0 * np.pi

    R = np.sqrt(sin_m**2 + cos_m**2)
    R = np.clip(R, 1e-6, 1.0)

    circ_std = np.sqrt(-2.0 * np.log(R))
    std_h = (circ_std / (2.0 * np.pi)) * 180.0

    return sin_m, cos_m, std_h


def compute_colour_features(img_rgb, mask_clean):
    mask_bool = mask_clean.astype(bool)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    fruit_pixels = hsv[mask_bool]

    if fruit_pixels.size == 0:
        raise ValueError("No foreground pixels found for HSV colour features.")

    H = fruit_pixels[:, 0]
    S = fruit_pixels[:, 1].astype(np.float32)
    V = fruit_pixels[:, 2].astype(np.float32)

    h_sin_mean, h_cos_mean, h_std = circular_mean_std(H)
    s_mean, v_mean = float(S.mean()), float(V.mean())
    s_std, v_std = float(S.std()), float(V.std())

    return np.array(
        [h_sin_mean, h_cos_mean, s_mean, v_mean, h_std, s_std, v_std],
        dtype=np.float32
    )


def build_feature_vector(shape_features, colour_features):
    geom = np.array([
        shape_features["area"],
        shape_features["perimeter"],
        shape_features["perimeter_norm"],
        shape_features["circularity"],
        shape_features["solidity"],
        shape_features["eccentricity"],
        shape_features["curvature"]
    ], dtype=np.float32)

    efd = np.array(shape_features["efd"], dtype=np.float32).flatten()
    radial = np.array(shape_features["radial_signature"], dtype=np.float32).flatten()
    zernike = np.array(shape_features["zernike"], dtype=np.float32).flatten()
    colour = np.array(colour_features, dtype=np.float32).flatten()

    return np.hstack([geom, efd, radial, zernike, colour])


def predict_single_image(image_path, model):
    try:
        img_rgb = load_and_normalize(image_path)
        raw_mask = segment_fruit(img_rgb)
        mask_refined = refine_mask(raw_mask)

        shape_feats = compute_shape_features(mask_refined)
        colour_feats = compute_colour_features(img_rgb, mask_refined)
        feat_vec = build_feature_vector(shape_feats, colour_feats)

        if np.isnan(feat_vec).any():
            return "vegetable"

        pred = model.predict([feat_vec])[0]
        return "fruit" if pred == 1 else "vegetable"
    except Exception:
        return "vegetable"


def predict_folder_to_csv(input_folder, output_csv, model):
    valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    rows = []

    for fname in sorted(os.listdir(input_folder)):
        if not fname.endswith(valid_exts):
            continue

        image_path = os.path.join(input_folder, fname)
        pred_label = predict_single_image(image_path, model)

        rows.append({
            "image": fname,
            "prediction": pred_label
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_folder> <output_csv>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_csv = sys.argv[2]

    model = joblib.load(MODEL_PATH)
    predict_folder_to_csv(input_folder, output_csv, model)


if __name__ == "__main__":
    main()