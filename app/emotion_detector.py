

import os
import pickle
import numpy as np
import cv2
from PIL import Image
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from django.conf import settings


class EmotionDetector:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.emotion_classes = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
        self.load_custom_model()
    
    def load_custom_model(self):
        try:
            model_path = os.path.join(settings.BASE_DIR, 'custom_emotion_model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'emotion_scaler.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Custom emotion model loaded successfully")
            else:
                print("No custom model found, using advanced heuristic detection")
        except Exception as e:
            print(f"Error loading custom model: {e}")
    
    def detect_emotion(self, image_cv):
        try:
            if self.model and self.scaler:
                result = self._detect_with_custom_model(image_cv)
                if result:
                    return result
            
            return self._detect_with_advanced_heuristics(image_cv)
            
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None
    
    def _detect_with_custom_model(self, image_cv):
        try:
            features = self._extract_emotion_features(image_cv)
            if features is None:
                return None
            
            features_scaled = self.scaler.transform([features])
            
            predicted_emotion = self.model.predict(features_scaled)[0]
            
            try:
                probabilities = self.model.predict_proba(features_scaled)[0]
                emotion_classes = self.model.classes_
                
                emotion_scores = {}
                for i, emotion in enumerate(emotion_classes):
                    emotion_scores[emotion] = round(probabilities[i], 3)
                
                for emotion in self.emotion_classes:
                    if emotion not in emotion_scores:
                        emotion_scores[emotion] = 0.001
                
                return {
                    'emotions': emotion_scores,
                    'dominant_emotion': predicted_emotion
                }
                
            except:
                emotion_scores = {emotion: 0.1 for emotion in self.emotion_classes}
                emotion_scores[predicted_emotion] = 0.7
                
                return {
                    'emotions': emotion_scores,
                    'dominant_emotion': predicted_emotion
                }
                
        except Exception as e:
            print(f"Error with custom model: {e}")
            return None
    
    def _detect_with_advanced_heuristics(self, image_cv):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
        
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        (x, y, w, h) = largest_face
        
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        
        emotion_scores = self._analyze_facial_patterns(face_normalized, face_roi)
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        return {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion
        }
    
    def _analyze_facial_patterns(self, face_normalized, face_original):
        emotion_scores = {
            'happy': 0.0,
            'sad': 0.0,
            'angry': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'neutral': 0.0
        }
        
        height, width = face_normalized.shape
        
        mouth_y_start = int(height * 0.65)
        mouth_y_end = int(height * 0.9)
        mouth_x_start = int(width * 0.25)
        mouth_x_end = int(width * 0.75)
        mouth_region = face_normalized[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
        
        if mouth_region.size > 0:
            mouth_bottom = mouth_region[-3:, :]
            mouth_corners = np.concatenate([mouth_region[:, :5], mouth_region[:, -5:]], axis=1)
            mouth_center = mouth_region[:, int(mouth_region.shape[1]*0.4):int(mouth_region.shape[1]*0.6)]
            
            if mouth_corners.size > 0 and mouth_center.size > 0:
                corner_brightness = np.mean(mouth_corners)
                center_brightness = np.mean(mouth_center)
                
                if corner_brightness > center_brightness + 0.05:
                    emotion_scores['happy'] += 0.6
                
                elif center_brightness > corner_brightness + 0.05:
                    emotion_scores['sad'] += 0.5
        
        eye_y_start = int(height * 0.25)
        eye_y_end = int(height * 0.45)
        eye_region = face_normalized[eye_y_start:eye_y_end, :]
        
        if eye_region.size > 0:
            eye_openness = np.std(eye_region)
            eye_brightness = np.mean(eye_region)
            
            if eye_openness > 0.15:
                emotion_scores['surprise'] += 0.5
            
            elif eye_openness < 0.08 and eye_brightness < 0.4:
                emotion_scores['angry'] += 0.4
                emotion_scores['disgust'] += 0.3
            
            elif eye_openness > 0.12 and eye_brightness > 0.5:
                emotion_scores['fear'] += 0.4
        
        brow_y_start = int(height * 0.15)
        brow_y_end = int(height * 0.35)
        brow_region = face_normalized[brow_y_start:brow_y_end, :]
        
        if brow_region.size > 0:
            brow_intensity = np.mean(brow_region)
            brow_variation = np.std(brow_region)
            
            if brow_intensity < 0.4 and brow_variation > 0.1:
                emotion_scores['angry'] += 0.4
            
            elif brow_intensity > 0.6:
                emotion_scores['surprise'] += 0.3
                emotion_scores['fear'] += 0.2
        
        face_brightness = np.mean(face_normalized)
        face_contrast = np.std(face_normalized)
        
        if face_contrast > 0.15:
            for emotion in ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']:
                if emotion_scores[emotion] > 0.2:
                    emotion_scores[emotion] += 0.2
        
        elif face_contrast < 0.1:
            emotion_scores['neutral'] += 0.4
        
        total_score = sum(emotion_scores.values())
        
        if total_score > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] = emotion_scores[emotion] / total_score
            
            dominant = max(emotion_scores, key=emotion_scores.get)
            if emotion_scores[dominant] > 0.3:
                boost = min(0.2, emotion_scores[dominant] * 0.5)
                emotion_scores[dominant] += boost
                
                remaining = 1.0 - emotion_scores[dominant]
                other_total = sum(emotion_scores[e] for e in emotion_scores if e != dominant)
                
                if other_total > 0:
                    for emotion in emotion_scores:
                        if emotion != dominant:
                            emotion_scores[emotion] = (emotion_scores[emotion] / other_total) * remaining
        else:
            emotion_scores['neutral'] = 0.7
            for emotion in ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']:
                emotion_scores[emotion] = 0.05
        
        total = sum(emotion_scores.values())
        for emotion in emotion_scores:
            emotion_scores[emotion] = round(emotion_scores[emotion] / total, 3)
        
        return emotion_scores
    
    def _extract_emotion_features(self, image_cv):
        """Extract comprehensive facial features for emotion detection"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            if len(faces) == 0:
                return None
            
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            (x, y, w, h) = largest_face
            face_roi = gray[y:y+h, x:x+w]
            
            face_resized = cv2.resize(face_roi, (96, 96))
            face_normalized = face_resized / 255.0
            
            features = []
            height, width = face_normalized.shape
            
            mouth_y_start = int(height * 0.65)
            mouth_y_end = int(height * 0.85)
            mouth_x_start = int(width * 0.3)
            mouth_x_end = int(width * 0.7)
            mouth_region = face_normalized[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end]
            
            if mouth_region.size > 0:
                mouth_top = mouth_region[:3, :]
                mouth_bottom = mouth_region[-3:, :]
                mouth_left = mouth_region[:, :5]
                mouth_right = mouth_region[:, -5:]
                mouth_center = mouth_region[:, int(mouth_region.shape[1]*0.4):int(mouth_region.shape[1]*0.6)]
                
                features.extend([
                    np.mean(mouth_left) - np.mean(mouth_center),
                    np.mean(mouth_right) - np.mean(mouth_center),
                    np.mean(mouth_top) - np.mean(mouth_bottom),
                    np.std(mouth_region),
                    np.mean(mouth_region),
                ])
                
                mouth_edges = cv2.Canny((mouth_region * 255).astype(np.uint8), 30, 100)
                mouth_width_pixels = np.sum(np.any(mouth_edges, axis=0))
                mouth_height_pixels = np.sum(np.any(mouth_edges, axis=1))
                features.extend([
                    mouth_width_pixels / mouth_region.shape[1],
                    mouth_height_pixels / mouth_region.shape[0],
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
            
            eye_y_start = int(height * 0.25)
            eye_y_end = int(height * 0.45)
            
            left_eye_region = face_normalized[eye_y_start:eye_y_end, int(width*0.2):int(width*0.45)]
            right_eye_region = face_normalized[eye_y_start:eye_y_end, int(width*0.55):int(width*0.8)]
            
            for eye_region in [left_eye_region, right_eye_region]:
                if eye_region.size > 0:
                    eye_top = eye_region[:int(eye_region.shape[0]*0.4), :]
                    eye_bottom = eye_region[int(eye_region.shape[0]*0.6):, :]
                    
                    features.extend([
                        np.mean(eye_top) - np.mean(eye_bottom),
                        np.std(eye_region),
                        np.var(eye_region),
                    ])
                else:
                    features.extend([0, 0, 0])
            
            brow_y_start = int(height * 0.15)
            brow_y_end = int(height * 0.35)
            
            left_brow = face_normalized[brow_y_start:brow_y_end, int(width*0.2):int(width*0.45)]
            right_brow = face_normalized[brow_y_start:brow_y_end, int(width*0.55):int(width*0.8)]
            
            for brow_region in [left_brow, right_brow]:
                if brow_region.size > 0:
                    brow_top = brow_region[:int(brow_region.shape[0]*0.5), :]
                    brow_bottom = brow_region[int(brow_region.shape[0]*0.5):, :]
                    
                    features.extend([
                        np.mean(brow_top) - np.mean(brow_bottom),
                        np.std(brow_region),
                        np.mean(brow_region),
                    ])
                else:
                    features.extend([0, 0, 0])
            
            face_aspect_ratio = height / width
            
            face_contrast = np.std(face_normalized)
            
            left_half = face_normalized[:, :width//2]
            right_half = cv2.flip(face_normalized[:, width//2:], 1)
            min_width = min(left_half.shape[1], right_half.shape[1])
            
            if min_width > 5:
                left_half = left_half[:, :min_width]
                right_half = right_half[:, :min_width]
                asymmetry = np.mean(np.abs(left_half - right_half))
            else:
                asymmetry = 0
            
            features.extend([
                face_aspect_ratio,
                face_contrast,
                asymmetry,
                np.mean(face_normalized),
            ])
            
            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def train_model(self, training_data):
        """Train emotion detection model with provided data"""
        try:
            X = []  # Features
            y = []  # Labels
            
            for emotion, images in training_data.items():
                for image_b64 in images:
                    try:
                        image_bytes = base64.b64decode(image_b64)
                        image = Image.open(io.BytesIO(image_bytes))
                        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        
                        features = self._extract_emotion_features(image_cv)
                        if features is not None:
                            X.append(features)
                            y.append(emotion)
                    except Exception as e:
                        print(f"Error processing image for {emotion}: {e}")
                        continue
            
            if len(X) < 5:
                return {'success': False, 'error': 'Not enough valid training samples'}
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            svm_model = SVC(kernel='rbf', probability=True, random_state=42)
            
            self.model = VotingClassifier(
                estimators=[('rf', rf_model), ('svm', svm_model)],
                voting='soft'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            model_path = os.path.join(settings.BASE_DIR, 'custom_emotion_model.pkl')
            scaler_path = os.path.join(settings.BASE_DIR, 'emotion_scaler.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            return {
                'success': True,
                'accuracy': accuracy,
                'samples_used': len(X),
                'message': f'Model trained with {accuracy*100:.1f}% accuracy on {len(X)} samples'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


