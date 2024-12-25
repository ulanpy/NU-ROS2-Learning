import pygame
import pickle
import numpy as np
import cv2 as cv

np.boo = np.bool_

# Initialize sound playback
pygame.mixer.init()
ulan_sound = pygame.mixer.Sound("src/name_sounds/ulan.mp3")
alya_sound = pygame.mixer.Sound("src/name_sounds/alya.mp3")


# Load saved embeddings, labels, SVM model, and label encoder
embeddings_data = np.load("models/faces_embeddings_miss1.npz")
embeddings = embeddings_data["embeddings"]
labels = embeddings_data["labels"]

with open("models/svm_model_miss1.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("models/label_encoder_miss1.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

# Initialize Haar cascade for face detection
haarcascade = cv.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Global variables for face tracking
MAX_DISTANCE = 80  # Max distance to match centroids between frames
lst = {"ulan": 0, "alya": 0}
SPEAK_INTERVAL = 5  # Minimum interval between sound plays in seconds
CONFIDENCE_THRESHOLD = 0.8
prev_frame_time = 0
new_frame_time = 0


# Frame skipping
PROCESS_INTERVAL = 60  # Process every 60th frame
