import numpy as np


def top_k_accuracy(y_true, y_pred, k=1):
    """From: https://github.com/chainer/chainer/issues/606

    Expects both y_true and y_pred to be one-hot encoded.
    """
    argsorted_y = np.argsort(y_pred)[:, -k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def image_distance(source_image, target_image, norm="L0"):
    if norm == "L0":
        distance = np.sum(source_image != target_image)
    elif norm == "L2":
        distance = np.sqrt(np.sum((source_image - target_image) ** 2))
    else:
        raise Exception(f"Norm {norm} is not supported")

    return distance
