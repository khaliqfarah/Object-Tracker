import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, process_noise=0.03, measurement_noise=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * process_noise
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)

def main():
    print("Starting main function")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    print("Video capture opened successfully")

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read video")
        return
    
    print("First frame read successfully")
    print("Please select the object to track by drawing a rectangle around it.")
    print("Then press SPACE or ENTER to confirm your selection.")
    
    # Select ROI
    try:
        bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
    except Exception as e:
        print(f"Error during object selection: {e}")
        return

    if bbox == (0, 0, 0, 0):
        print("No object selected. Exiting.")
        return

    print(f"Object selected: {bbox}")

    # Initialize tracker
    try:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return

    print("Tracker initialized successfully")

    kf = KalmanFilter()
    trajectory = []
    max_trajectory_points = 20

    print("Entering main loop")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        # Update tracker
        try:
            success, bbox = tracker.update(frame)
        except Exception as e:
            print(f"Error updating tracker: {e}")
            break

        if success:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

            # Get center point
            center = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
            
            # Kalman filter correction and prediction
            measurement = np.array([[np.float32(center[0])], [np.float32(center[1])]])
            kf.correct(measurement)
            prediction = kf.predict()
            predicted_position = (int(prediction[0]), int(prediction[1]))
            
            # Update trajectory
            trajectory.append(predicted_position)
            if len(trajectory) > max_trajectory_points:
                trajectory.pop(0)

            # Draw current position
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            
            # Draw predicted position
            cv2.circle(frame, predicted_position, 5, (255, 0, 0), -1)
            
            # Draw trajectory
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i-1], trajectory[i], (0, 0, 255), 2)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Precise Object Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Exiting main loop")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Script started")
    try:
        main()
    except Exception as e:
        print(f"An error occurred in the main function: {e}")
    print("Script ended")
