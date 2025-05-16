# from ..extractor import geomtry_extractor
import cv2
import numpy as np

# class geomtry_classifier:


def classify_by_aspect_ratio():
    pass


def classify_by_number_pins():
    pass


def classify_by_position_pins():
    pass


def classify_by_geomtry():
    classify_by_position_pins()
    classify_by_number_pins()
    classify_by_aspect_ratio()
    pass

# @classmethod


def classify_knn(candidate_hist, hog_dict, k=3):
    """
    Classify a candidate histogram based on KNN using cv2.compareHist.

    Parameters:
        candidate_hist (numpy.ndarray): Histogram (as a 1D array of float32) of the candidate image.
        k (int): Number of nearest neighbors to consider.

    Returns:
        label (str): The predicted class label.
    """
    # List to hold tuples of (similarity, label)
    neighbors = []

    # Loop over each class in the global hog_dict
    for label, records in hog_dict.items():
        for rec in records:
            # Get the stored hog vector and convert to np.float32
            record_hist = np.array(rec['hog_vector'], dtype=np.float32)
            # Ensure the candidate and record have the same shape
            # print("Candidate shape:", candidate_hist.shape)
            # print("Record shape:", record_hist.shape)
            if record_hist.shape != candidate_hist.shape:
                continue
            # Use correlation metric; higher values mean more similar
            # Using histogram intersection as similarity measure
            candidate_hist_flat = candidate_hist.flatten()
            record_hist_flat = record_hist.flatten()
            dot_product = np.dot(candidate_hist_flat, record_hist_flat)
            norm_candidate = np.linalg.norm(candidate_hist_flat)
            norm_record = np.linalg.norm(record_hist_flat)
            similarity = dot_product / (norm_candidate * norm_record) if norm_candidate and norm_record else 0
            neighbors.append((similarity, label))
    # Sort by similarity in descending order (higher correlation is better)
    neighbors.sort(key=lambda x: x[0], reverse=True)

    # Select top k neighbors
    top_neighbors = neighbors[:k]
    # print("Top neighbors:", top_neighbors)
    # Use majority vote to classify
    votes = {}
    for sim, vote_label in top_neighbors:
        votes[vote_label] = votes.get(vote_label, 0) + 1

    # Check if no neighbors were found
    if not votes:
        return "Unknown"

    # Find the label(s) with the maximum vote count
    max_votes = max(votes.values())
    candidate_labels = [lbl for lbl,
                        count in votes.items() if count == max_votes]
    # If there's a tie, pick the one with highest total similarity score
    if len(candidate_labels) > 1:
        sim_sums = {lbl: 0 for lbl in candidate_labels}
        for sim, vote_label in top_neighbors:
            if vote_label in sim_sums:
                sim_sums[vote_label] += sim
        predicted_label = max(sim_sums.items(), key=lambda x: x[1])[0]
    else:
        predicted_label = candidate_labels[0]

    return predicted_label
