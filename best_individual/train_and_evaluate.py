import warnings

warnings.filterwarnings("ignore")

import os
import cv2
import joblib
import mahotas
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pyefd import elliptic_fourier_descriptors
from skimage import morphology, measure
from skimage.filters import threshold_otsu

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ===============================
# Config
# ===============================
BASE_DIR = "."
LABEL_PATH = "./labels.xlsx"
RESULT_DIR = "./results"
MODEL_PATH = "./results/weighted_ensemble_model.pkl"

TARGET_SIZE = (90, 90)
CLASS_COL = "food_name"

os.makedirs(RESULT_DIR, exist_ok=True)


# ===============================
# Step 1: Image preprocessing
# ===============================
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


# ===============================
# Step 2: Segmentation
# ===============================
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


# ===============================
# Step 3: Mask refinement
# ===============================
def refine_mask(
    mask: np.ndarray,
    hole_area: int = 200,
    min_object_size: int = 300,
    closing_radius: int = 5,
    opening_radius: int = 2,
) -> np.ndarray:
    mask_bool = mask > 0

    mask_filled = morphology.remove_small_holes(
        mask_bool,
        area_threshold=hole_area
    )

    mask_clean = morphology.remove_small_objects(
        mask_filled,
        min_size=min_object_size
    )

    if closing_radius > 0:
        mask_clean = morphology.closing(
            mask_clean,
            morphology.disk(closing_radius)
        )

    if opening_radius > 0:
        mask_clean = morphology.opening(
            mask_clean,
            morphology.disk(opening_radius)
        )

    labeled = measure.label(mask_clean)
    if labeled.max() == 0:
        return mask_filled.astype(np.uint8) * 255

    regions = measure.regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    mask_final = labeled == largest.label
    return mask_final.astype(np.uint8) * 255


# ===============================
# Step 4: Feature extraction
# ===============================
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


def extract_feature_row(row):
    img_path = row["image_path"]

    img_rgb = load_and_normalize(img_path)
    raw_mask = segment_fruit(img_rgb)
    mask_refined = refine_mask(raw_mask)

    shape_feats = compute_shape_features(mask_refined)
    colour_feats = compute_colour_features(img_rgb, mask_refined)
    feature_vector = build_feature_vector(shape_feats, colour_feats)

    if np.isnan(feature_vector).any():
        raise ValueError("Feature vector contains NaN")

    out = {
        "image_id": row["image_id"],
        "label": str(row["label"]).strip(),
        "image_path": img_path
    }

    for j, val in enumerate(feature_vector):
        out[f"f_{j}"] = float(val)

    return out


def extract_features_from_df(input_df, df_name="dataset"):
    all_rows = []
    failed_rows = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"Extracting {df_name}"):
        try:
            out_row = extract_feature_row(row)
            all_rows.append(out_row)
        except Exception as e:
            failed_rows.append({
                "image_id": row["image_id"],
                "label": row["label"],
                "image_path": row["image_path"],
                "error": str(e)
            })

    return pd.DataFrame(all_rows), pd.DataFrame(failed_rows)


# ===============================
# Ensemble helper functions
# ===============================
def weighted_ensemble_predict_proba(models_dict, weights_dict, X, class_order):
    n = len(X)
    n_classes = len(class_order)
    proba_sum = np.zeros((n, n_classes), dtype=float)

    for name, model in models_dict.items():
        w = weights_dict[name]
        proba = model.predict_proba(X)

        model_classes = list(model.classes_)
        aligned = np.zeros((n, n_classes), dtype=float)

        for j, cls in enumerate(model_classes):
            idx = class_order.index(cls)
            aligned[:, idx] = proba[:, j]

        proba_sum += w * aligned

    return proba_sum


def weighted_ensemble_predict(models_dict, weights_dict, X, class_order):
    proba = weighted_ensemble_predict_proba(models_dict, weights_dict, X, class_order)
    pred_idx = np.argmax(proba, axis=1)
    return np.array([class_order[i] for i in pred_idx])


# ===============================
# Step 5: Data preparation
# ===============================
print("Loading labels...")
df = pd.read_excel(LABEL_PATH)
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())

if CLASS_COL not in df.columns:
    raise ValueError(f"Column '{CLASS_COL}' not found in labels.xlsx.")

extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
image_dict = {}

for root, _, files in os.walk(BASE_DIR):
    for f in files:
        name, ext = os.path.splitext(f)
        if ext in extensions:
            image_dict[name] = os.path.join(root, f)

print(f"Found {len(image_dict)} images in dataset.")

success = 0
missing = 0

for _, row in df.iterrows():
    img_id = str(row["image_id"]).strip()
    img_path = image_dict.get(img_id)

    if img_path is None:
        missing += 1
        continue

    try:
        _ = Image.open(img_path)
        success += 1
    except Exception:
        missing += 1

print("\n====== Summary ======")
print("Readable images:", success)
print("Failed / missing:", missing)

df["image_path"] = df["image_id"].astype(str).str.strip().map(image_dict)

missing_mask = df["image_path"].isna()
if missing_mask.any():
    print("Missing image_id examples:")
    print(df.loc[missing_mask, "image_id"].head())

df = df.loc[~missing_mask].reset_index(drop=True)

# 关键修改：统一成 4 类
df["label"] = df[CLASS_COL].astype(str).str.strip().str.lower()

print("\nUnique classes:")
print(sorted(df["label"].unique().tolist()))
print("Number of classes:", df["label"].nunique())

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTotal samples: {len(df)}")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")


# ===============================
# Step 6: Feature extraction
# ===============================
train_features_df, train_failed_df = extract_features_from_df(train_df, df_name="train")
test_features_df, test_failed_df = extract_features_from_df(test_df, df_name="test")

print("\n===== Train Summary =====")
print("Successful rows:", len(train_features_df))
print("Failed rows:", len(train_failed_df))

print("\n===== Test Summary =====")
print("Successful rows:", len(test_features_df))
print("Failed rows:", len(test_failed_df))

feature_cols = [c for c in train_features_df.columns if c.startswith("f_")]

train_features_clean = train_features_df.dropna(subset=feature_cols).reset_index(drop=True)
test_features_clean = test_features_df.dropna(subset=feature_cols).reset_index(drop=True)

print("\nAfter dropping NaN rows:")
print("Train rows:", len(train_features_clean))
print("Test rows:", len(test_features_clean))


# ===============================
# Step 7: Prepare training matrices
# ===============================
X_train = train_features_clean[feature_cols].copy()
y_train = train_features_clean["label"].copy()

X_test = test_features_clean[feature_cols].copy()
y_test = test_features_clean["label"].copy()

# 关键修改：只保留 RF + SVM
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}


# ===============================
# Step 8: Single-model held-out evaluation
# ===============================
results = []

for name, model in models.items():
    model_fit = clone(model)
    model_fit.fit(X_train, y_train)
    y_pred = model_fit.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "model": name,
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1
    })

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y_train.unique())))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
print("\n===== Model Comparison on Held-out Test Set =====")
print(results_df)


# ===============================
# Step 9: Cross-validation and ensemble weights
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro"
}
cv_results = {}

for name, model in models.items():
    scores = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    cv_results[name] = {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "precision_macro_mean": scores["test_precision_macro"].mean(),
        "precision_macro_std": scores["test_precision_macro"].std(),
        "recall_macro_mean": scores["test_recall_macro"].mean(),
        "recall_macro_std": scores["test_recall_macro"].std(),
        "f1_macro_mean": scores["test_f1_macro"].mean(),
        "f1_macro_std": scores["test_f1_macro"].std()
    }

cv_results_df = pd.DataFrame(cv_results).T.sort_values("f1_macro_mean", ascending=False)
print("\n===== Cross Validation Results =====")
print(cv_results_df.round(6))

f1_means = cv_results_df["f1_macro_mean"].to_dict()
f1_sum = sum(f1_means.values())

if f1_sum <= 0:
    raise ValueError("Sum of CV F1 scores is non-positive; cannot compute ensemble weights.")

ensemble_weights = {name: val / f1_sum for name, val in f1_means.items()}

print("\n===== Ensemble Weights (normalized CV F1-macro) =====")
for name, w in ensemble_weights.items():
    print(f"{name}: {w:.6f}")


# ===============================
# Step 10: Retrain all models on full training set and save ensemble
# ===============================
fitted_models = {}

for name, model in models.items():
    model_fit = clone(model)
    model_fit.fit(X_train, y_train)
    fitted_models[name] = model_fit

class_order = sorted(y_train.unique().tolist())

ensemble_artifact = {
    "models": fitted_models,
    "weights": ensemble_weights,
    "class_order": class_order,
    "feature_names": feature_cols
}

joblib.dump(ensemble_artifact, MODEL_PATH)

print("\nSaved weighted ensemble model to:", MODEL_PATH)
print("Class order:", class_order)


# ===============================
# Step 11: Evaluate weighted ensemble on held-out test set
# ===============================
y_pred_ens = weighted_ensemble_predict(
    models_dict=fitted_models,
    weights_dict=ensemble_weights,
    X=X_test,
    class_order=class_order
)

acc_ens = accuracy_score(y_test, y_pred_ens)
prec_ens = precision_score(y_test, y_pred_ens, average="macro", zero_division=0)
rec_ens = recall_score(y_test, y_pred_ens, average="macro", zero_division=0)
f1_ens = f1_score(y_test, y_pred_ens, average="macro", zero_division=0)

print("\n" + "=" * 60)
print("Weighted Ensemble")
print("=" * 60)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_ens, labels=class_order))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ens, zero_division=0))

print("\nWeighted Ensemble Metrics:")
print({
    "accuracy": round(acc_ens, 6),
    "precision_macro": round(prec_ens, 6),
    "recall_macro": round(rec_ens, 6),
    "f1_macro": round(f1_ens, 6)
})


# ===============================
# Step 12: Generate CSV on held-out test set using ensemble
# ===============================
def predict_single_image(image_path, ensemble_obj, default_label=None):
    try:
        img_rgb = load_and_normalize(image_path)
        raw_mask = segment_fruit(img_rgb)
        mask_refined = refine_mask(raw_mask)

        shape_feats = compute_shape_features(mask_refined)
        colour_feats = compute_colour_features(img_rgb, mask_refined)
        feat_vec = build_feature_vector(shape_feats, colour_feats)

        if np.isnan(feat_vec).any():
            return default_label if default_label is not None else "unknown"

        feature_names = ensemble_obj["feature_names"]
        class_order_local = ensemble_obj["class_order"]
        models_dict = ensemble_obj["models"]
        weights_dict = ensemble_obj["weights"]

        X_one = pd.DataFrame([feat_vec], columns=feature_names)
        pred = weighted_ensemble_predict(
            models_dict=models_dict,
            weights_dict=weights_dict,
            X=X_one,
            class_order=class_order_local
        )[0]

        return str(pred)

    except Exception:
        return default_label if default_label is not None else "unknown"


def predict_df_to_csv(input_df, output_csv, ensemble_obj, default_label=None):
    rows = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Predicting images"):
        pred_label = predict_single_image(
            row["image_path"],
            ensemble_obj=ensemble_obj,
            default_label=default_label
        )
        rows.append({
            "image_id": row["image_id"],
            "prediction": pred_label
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")
    return pred_df


majority_label = y_train.value_counts().idxmax()
test_csv_path = os.path.join(RESULT_DIR, "test_predictions_weighted_ensemble.csv")
test_pred_df = predict_df_to_csv(
    test_df,
    test_csv_path,
    ensemble_obj=ensemble_artifact,
    default_label=majority_label
)
print(test_pred_df.head())