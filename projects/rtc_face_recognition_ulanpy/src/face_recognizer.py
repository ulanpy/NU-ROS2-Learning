from keras_facenet import FaceNet
import time
from src.__init__ import *
from typing import Optional
from numpy import ndarray


class FaceRecognizer:
    def __init__(self, used_model, used_encoder, last_spoken_time: dict) -> None:
        self.model = used_model
        self.encoder = used_encoder
        self.facenet = FaceNet()
        self.last_spoken_time = last_spoken_time
        self.tracked_faces = {}
        self.frame_count = 0
        self.new_frame_time = 0
        self.prev_frame_time = 0
        self.updated_tracked_faces = {}
        self.current_centroids = []  # List to store centroids for detected faces
        self.name = ""
        self.label = ""
        self.matched_face = None
        self.rgb_img = None
        self.gray_img = None
        self.fps = 0
        self.centroid = []

    def get_embedding(self, face_img: ndarray) -> ndarray:
        face_img = face_img.astype("float32")
        face_img = np.expand_dims(face_img, axis=0)
        return self.facenet.embeddings(face_img)[0]

    def process_new_frame(self, frame: ndarray) -> tuple[int, ndarray]:
        self.current_centroids = []
        frame_count = self.get_frame_count()
        self.set_rgb_and_gray_channels(frame)
        faces = self.get_face_box()
        return frame_count, faces

    def set_rgb_and_gray_channels(self, frame: ndarray) -> None:
        self.rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def get_face_box(self) -> ndarray:
        return haarcascade.detectMultiScale(self.gray_img, 1.3, 5)

    def get_frame_count(self) -> int:
        self.frame_count += 1
        # FPS calculation
        self.new_frame_time = time.time()
        self.fps = 1 / (self.new_frame_time - self.prev_frame_time)
        # FPS is 1 divided by the time difference between frames
        self.prev_frame_time = self.new_frame_time
        return self.frame_count

    def process_new_centroid(self, x: int, y: int, w: int, h: int):
        self.centroid = (int(x + w / 2), int(y + h / 2))
        self.current_centroids.append(self.centroid)
        return self.centroid

    def predict_face(self, x: int, y: int, w: int, h: int) -> tuple[str, str]:
        img = cv.resize(self.rgb_img[y:y + h, x:x + w], (160, 160))
        # Extract face and compute embedding
        embedding = self.get_embedding(img)
        # Predict using SVM model
        probabilities = self.model.predict_proba([embedding])[0]
        max_index = np.argmax(probabilities)
        max_probability = probabilities[max_index]

        if max_probability >= CONFIDENCE_THRESHOLD:
            self.name = self.encoder.inverse_transform([max_index])[0]
            self.label = f"{self.name} ({max_probability:.2f})"
        else:
            self.name = "unknown"
            most_probable_name = self.encoder.inverse_transform([max_index])[0]
            type(most_probable_name)
            self.label = f"unknown"
        return self.name, self.label

    def update_tracked_faces(self, face_centroid: tuple, name: str, label: str) -> None:
        # Update tracked faces or match with previously tracked faces
        self.matched_face = self.match_face_with_centroid(face_centroid)
        if self.matched_face is None:
            self.updated_tracked_faces[face_centroid] = (name, label)
        else:
            self.updated_tracked_faces[face_centroid] = self.tracked_faces[self.matched_face]

        self.tracked_faces = self.updated_tracked_faces

    def process_face(self, x, y, w, h) -> tuple[str, str]:
        face_centroid = self.process_new_centroid(x, y, w, h)
        name, label = self.predict_face(x, y, w, h)
        self.update_tracked_faces(face_centroid, name, label)
        self.play_sound()
        return name, label

    def match_face_with_centroid(self, centroid: tuple) -> Optional[tuple]:
        # Ensure the current centroid is valid
        if self.current_centroids is None:
            print("Warning: Current centroid is None. Skipping match_face.")
            return None

        # Match current face with previously tracked faces
        matched_face_centroid = None
        min_distance = float("inf")
        for prev_centroid, prev_data in self.tracked_faces.items():
            if prev_centroid is None:
                continue  # Skip invalid centroids
            distance = self.euclidean_distance(centroid, prev_centroid)
            if distance < MAX_DISTANCE and distance < min_distance:
                matched_face_centroid = prev_centroid
                min_distance = distance
        return matched_face_centroid

    @staticmethod
    def euclidean_distance(centroid: tuple, prev_centroid: tuple) -> float:
        return np.sqrt((centroid[0] - prev_centroid[0]) ** 2 + (centroid[1] - prev_centroid[1]) ** 2)

    def play_sound(self) -> None:
        current_time = time.time()
        if self.name == "ulan" and (current_time - self.last_spoken_time["ulan"]) > SPEAK_INTERVAL:
            pygame.mixer.Sound.play(ulan_sound)
            self.last_spoken_time["ulan"] = int(current_time)
        elif self.name == "alya" and (current_time - self.last_spoken_time["alya"]) > SPEAK_INTERVAL:
            pygame.mixer.Sound.play(alya_sound)
            self.last_spoken_time["alya"] = int(current_time)

    def process_faces(self, frame, faces):
        """
        Processes detected faces and modifies the frame accordingly.
        """
        self.updated_tracked_faces = {}  # Reset updated faces
        return self._process_and_modify_frame(frame, faces, process_method=self.process_face)

    def track_faces(self, frame, faces):
        """
        Tracks detected faces and modifies the frame based on existing tracked data.
        """
        return self._process_and_modify_frame(frame, faces, process_method=self._track_face_helper)

    def _process_and_modify_frame(self, frame, faces, process_method):
        """
        A generic method to process faces and modify the frame based on the given processing method.
        """
        modified_frame = frame
        for x, y, w, h in faces:
            try:
                name, label = process_method(x, y, w, h)
                modified_frame = self.modified_frame(name, label, frame, self.fps, x, y, w, h)
            except Exception as e:
                print(f"Error processing face: {e}")
        return modified_frame

    def _track_face_helper(self, x, y, w, h):
        """
        Helper function for tracking faces.
        """
        centroid = self.process_new_centroid(x, y, w, h)
        matched_face_centroid = self.match_face_with_centroid(centroid)
        if matched_face_centroid is not None:
            return self.tracked_faces[matched_face_centroid]
        return "Unknown", "Analyzing..."

    @staticmethod
    def modified_frame(name: str, label: str, frame: ndarray, fps: int, x: int, y: int, w: int, h: int) -> ndarray:
        # Draw rectangle and label on the current frame
        if label == "Analyzing...":
            color = (0, 255, 255)  # Yellow color for "Analyzing..."
        elif name != "unknown":
            color = (0, 255, 0)  # Green for recognized faces
        else:
            color = (0, 0, 255)  # Red for unknown faces
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, str(label), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        return frame
