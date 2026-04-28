import yaml
import itertools
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, hamming_loss, f1_score


# -------------------------
# Config Loader
# -------------------------
def load_config(path="configs/params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Label Processing
# -------------------------
def process_labels(series):
    """
    Convert space-separated label string into list
    Example: "cs.lg cs.ai" → ["cs.lg", "cs.ai"]
    """
    return series.apply(lambda x: str(x).split())


# -------------------------
# MultiLabel Binarizer
# -------------------------
def fit_mlb(y_train_labels):
    """
    Fit MultiLabelBinarizer on training labels
    """
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train_labels)
    return mlb, y_train


def transform_mlb(mlb, y_labels):
    """
    Transform labels using fitted MLB
    """
    return mlb.transform(y_labels)


# -------------------------
# Parameter Grid Generator
# -------------------------
def generate_param_grid(search_space):
    """
    Generate combinations of hyperparameters
    """
    keys = list(search_space.keys())
    values = list(search_space.values())

    return [
        dict(zip(keys, v))
        for v in itertools.product(*values)
    ]


# -------------------------
# Thresholding
# -------------------------
def apply_threshold(probs, threshold):
    """
    Convert probabilities to binary predictions
    """
    return (probs > threshold).astype(int)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(y_true, y_pred):
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_micro": precision_score(y_true, y_pred, average="micro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_micro": recall_score(y_true, y_pred, average="micro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


# -------------------------
# ClassifierChain Prob Fix
# -------------------------
def chain_predict_proba_to_array(probs_list):
    """
    Convert list of probability arrays (ClassifierChain)
    into (n_samples, n_labels) array
    """
    return np.array([p[:, 1] for p in probs_list]).T