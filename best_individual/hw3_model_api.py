import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "results", "weighted_ensemble_model.pkl")

_ensemble = None


def load_model():
    global _ensemble
    if _ensemble is None:
        _ensemble = joblib.load(MODEL_PATH)
    return _ensemble


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


def predict(feature_vector):
    ensemble = load_model()

    feature_names = ensemble["feature_names"]
    class_order = ensemble["class_order"]
    models_dict = ensemble["models"]
    weights_dict = ensemble["weights"]

    x = np.asarray(feature_vector, dtype=float)

    if x.ndim == 1:
        if x.shape[0] != len(feature_names):
            raise ValueError(
                f"Expected feature vector of length {len(feature_names)}, got {x.shape[0]}."
            )
        x = x.reshape(1, -1)
    elif x.ndim == 2 and x.shape[0] == 1:
        if x.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected feature vector of length {len(feature_names)}, got {x.shape[1]}."
            )
    else:
        raise ValueError("feature_vector must be a 1D vector or shape (1, n_features).")

    x_df = pd.DataFrame(x, columns=feature_names)
    proba = weighted_ensemble_predict_proba(models_dict, weights_dict, x_df, class_order)
    pred_idx = int(np.argmax(proba[0]))
    return str(class_order[pred_idx])