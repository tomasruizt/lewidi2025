import numpy as np
import ot


def wasserstein_distance(target, prediction):
    target = np.array(target)
    prediction = np.array(prediction)
    size = len(target)
    optimal_transport_matrix = np.abs(np.arange(size).reshape(-1, 1) - np.arange(size))
    dist_wass = ot.emd2(target, prediction, optimal_transport_matrix)
    return dist_wass


def average_WS(targets, predictions):
    """
    Calculates the average Wasserstein distance (average WD) between
    corresponding pairs of target and prediction distributions.
    Parameters:
      - targets (list of lists): A list of target distributions.
      - predictions (list of lists): A list of predicted distributions.
    Returns:
        - float: averafe Wasserstein distance
    """
    distances = [wasserstein_distance(p, t) for p, t in zip(targets, predictions)]
    average_distance = round(sum(distances) / len(targets), 5)
    return average_distance
