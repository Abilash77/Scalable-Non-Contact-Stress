import cv2
import time
from deepface import DeepFace

def main():
    print("[INFO] Initializing Real-Time Face Stress Detector...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    # Use Haar Cascade for fast face detection to draw boxes
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("[INFO] Webcam opened successfully. Press 'q' to quit.")

    # Frame processing rate limit
    process_this_frame = True
    stress_status = "Unknown"
    stress_color = (255, 255, 255)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        # We only process every other frame to keep it real-time
        if process_this_frame:
            try:
                # Detect emotion using DeepFace on the small frame
                result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, silent=True)
                
                if isinstance(result, list):
                    result = result[0]
                
                dominant_emotion = result.get('dominant_emotion', 'neutral')
                
                # Map emotions to Stress
                # High stress: angry, disgust, fear, sad
                # Low stress: happy, neutral, surprise
                if dominant_emotion in ['angry', 'disgust', 'fear', 'sad']:
                    stress_status = f"Stressed ({dominant_emotion})"
                    stress_color = (0, 0, 255) # Red
                else:
                    stress_status = f"Not Stressed ({dominant_emotion})"
                    stress_color = (0, 255, 0) # Green
            except Exception as e:
                pass
                
        process_this_frame = not process_this_frame

        # Draw a box around faces in the original frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), stress_color, 2)
            cv2.putText(frame, stress_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stress_color, 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Face Stress Detection', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")

if __name__ == '__main__':
    main()
