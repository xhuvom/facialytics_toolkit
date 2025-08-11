# utils/face_comparison.py
import os
import cv2
import numpy as np
from utils.models import COMPARISON_MODEL

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[Comparison] ERROR: Could not read image: {image_path}")
        return None
    faces = COMPARISON_MODEL.get(img)
    if not faces:
        print(f"[Comparison] No face detected in image: {image_path}")
        return None
    # Get the face with the highest detection score
    main_face = max(faces, key=lambda x: x.det_score)
    embedding = main_face.embedding
    norm_embedding = embedding / np.linalg.norm(embedding)
    return norm_embedding

def compare_faces(source_image_path, target_image_path, threshold=65):
    print(f"[Comparison] Comparing {source_image_path} and {target_image_path}")
    source_embedding = get_face_embedding(source_image_path)
    target_embedding = get_face_embedding(target_image_path)
    if source_embedding is None or target_embedding is None:
        return None, "Face not detected in one of the images."
    similarity = np.dot(source_embedding, target_embedding)
    similarity = max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
    similarity_percent = round(similarity * 100, 2)
    verdict = "Same Person" if similarity_percent >= threshold else "Different Person"
    print(f"[Comparison] Similarity: {similarity_percent}%. Verdict: {verdict}")
    return similarity_percent, verdict

