import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import ffmpeg
import shutil
class KalmanFilter(object):
    def __init__(self, dt, INIT_POS_STD, INIT_VEL_STD, ACCEL_STD, GPS_POS_STD):

        """
        :param dt: sampling time (time for 1 cycle)
        :param INIT_POS_STD: initial position standard deviation in x-direction
        :param INIT_VEL_STD: initial position standard deviation in y-direction
        :param ACCEL_STD: process noise magnitude
        :param GPS_POS_STD: standard deviation of the measurement
        """

        # Define sampling time
        self.dt = dt

        # Intial State
        self.x = np.zeros((4, 1))

        # State Estimate Covariance Matrix
        cov = np.zeros((4, 4))
        cov[0, 0] = INIT_POS_STD ** 2
        cov[1, 1] = INIT_POS_STD ** 2
        cov[2, 2] = INIT_VEL_STD ** 2
        cov[3, 3] = INIT_VEL_STD ** 2
        self.P = cov

        # State Transition Matrix
        self.F = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Covariance of Process model noise
        q = np.zeros((2, 2))
        q[0, 0] = ACCEL_STD ** 2
        q[1, 1] = ACCEL_STD ** 2
        self.q = q

        # Process Model Sensitivity Matrix
        L = np.zeros((4, 2))
        L[0, 0] = 0.5 * self.dt ** 2
        L[1, 1] = 0.5 * self.dt ** 2
        L[2, 0] = self.dt
        L[3, 1] = self.dt
        self.L = L

        # Process model noise
        self.Q = np.dot(self.L, np.dot(self.q, (self.L).T))

        # Define Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Measurement Covariance Matrix
        R = np.zeros((2, 2))
        R[0, 0] = GPS_POS_STD ** 2
        R[1, 1] = GPS_POS_STD ** 2
        self.R = R

    # PREDICTION STEP
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, (self.F).T)) + self.Q

        x_pred = self.x[0]
        y_pred = self.x[1]
        return x_pred, y_pred

    # UPDATE STEP
    def update(self, z):
        # Innovation
        z_hat = np.dot(self.H, self.x)
        self.y = z - z_hat

        # Innovation Covariance
        self.S = np.dot(self.H, np.dot(self.P, (self.H).T)) + self.R

        # Kalman Gain
        self.K = np.dot(self.P, np.dot((self.H).T, np.linalg.inv(self.S)))

        I = np.eye(4)

        self.x = self.x + np.dot(self.K, self.y)
        self.P = np.dot((I - np.dot(self.K, self.H)), self.P)

        x_updated = self.x[0]
        y_updated = self.x[1]
        return x_updated, y_updated

def get_bounding_box_center_frame(frame, model, names, object_class):
    centers = []
    results = model(frame)
    person_detected = False

    for result in results:
        detections = []
        
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            class_name = names.get(class_id)

            if score > 0.2:
                if not person_detected:
                    person_detected = True
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    centers.append((center_x, center_y))

    if not person_detected:
        centers.append("Not detected")
    # print(result.boxes.data.tolist())
    return centers


def create_video(frames_patten='Track/%03d.png', video_file = 'movie.mp4', framerate=25):
  if os.path.exists(video_file):
      os.remove(video_file)
  ffmpeg.input(frames_patten, framerate=framerate).output(video_file).run()


def main():
    # Get the current directory
    current_directory = os.getcwd()
    print(current_directory)

    # Set input and output directory
    video_path = os.path.join(current_directory,'running.mp4')
    output_video_path = os.path.join(current_directory, 'Output', 'running_4_kf.mp4')
    print(video_path)

    # Instantiate model
    weights_path = './yolov8n.pt'# './datasets/runs/detect/train9/weights/best.pt'
    print(weights_path)
    model = YOLO(weights_path)
    names = model.names
    print(names)

    folder_path = "frames"
    if os.path.exists(folder_path):  # Check if folder exists
        print(f"Folder '{folder_path}' exists. Removing...")
        try:
            shutil.rmtree(folder_path)  # Recursively remove folder and its contents
            print(f"Folder '{folder_path}' successfully removed.")
        except Exception as e:
            print(f"Error: Failed to remove folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' does not exist.")
    os.makedirs(folder_path)
    # Kalman filter parameters
    dt = 1/30  # Sampling time = FPS
    INIT_POS_STD = 10  # Initial position standard deviation
    INIT_VEL_STD = 10  # Initial velocity standard deviation
    ACCEL_STD = 40  # Acceleration standard deviation
    GPS_POS_STD = 1  # Measurement position standard deviation

    # Kalman filter initialization
    kf = KalmanFilter(dt, INIT_POS_STD, INIT_VEL_STD, ACCEL_STD, GPS_POS_STD)


    # Open the video file
    cap = cv2.VideoCapture(video_path)
    isFirstFrame = True

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    i = 0

    while True:
        # Read frame from the video
        ret, frame = cap.read()
        if i == 500:
            break
        print()
        # Break the loop if there are no more frames to read
        if not ret:
            break

        # Create the legend circles
        true_circle_position = (20, 20)
        predict_circle_position = (20, 50)
        update_circle_position = (20, 80)
        circle_radius = 6
        true_circle_color = (0, 255, 0)  # Green color for data
        predict_circle_color = (255, 0, 0)  # Blue color for forecast
        update_circle_color = (0, 0, 255)  # Red color for forecast
        circle_thickness = 2  # Filled circle

        # Draw the legend circles
        cv2.circle(frame, true_circle_position, circle_radius, true_circle_color, circle_thickness)
        cv2.circle(frame, predict_circle_position, circle_radius, predict_circle_color, circle_thickness)
        cv2.circle(frame, update_circle_position, circle_radius, update_circle_color, circle_thickness)

        # Draw the legend
        cv2.putText(frame, "True", (40, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, true_circle_color, 2)
        cv2.putText(frame, "Predict", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, .5, predict_circle_color, 2)
        cv2.putText(frame, "Update", (40, 85), cv2.FONT_HERSHEY_SIMPLEX, .5, update_circle_color, 2)

        # Process the frame to get bounding box centers
        centers = get_bounding_box_center_frame(frame, model, names, object_class='ferrari')
        print("centers: ", centers)
        # Check if center is detected
        if len(centers) > 0:
            center = centers[0]  # Extract the first center tuple

            # Example: Draw circle at the center
            if isinstance(center, tuple):
                print("Center = ", center)
                cv2.circle(frame, center, radius=20, color=(0, 255, 0), thickness=4) # Green

                x_pred, y_pred = kf.predict()
                if isFirstFrame:  # First frame
                    x_pred = round(x_pred[0])
                    y_pred = round(y_pred[0])
                    print("Predicted: ", (x_pred, y_pred))
                    isFirstFrame = False
                else:
                    x_pred = round(x_pred[0])
                    y_pred = round(y_pred[1])
                    print("Predicted: ", (x_pred, y_pred))

                cv2.circle(frame, (x_pred, y_pred), radius=20, color=(255, 0, 0), thickness=4) #  Blue

                # Update
                (x1, y1) = kf.update(center)
                x_updt = round(x1[0])
                y_updt =  round(x1[1])
                print("Update: ", (x_updt, y_updt))
                cv2.circle(frame, (x_updt, y_updt), radius=20, color= (0, 0, 255), thickness=4) # Red

        # Write frame to the output video
        out.write(frame)

        # Display the frame with circles
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imsave(f"./frames/Frame_{i:03d}.png", f)
        print(f"saving {i}")
        print("saving file")
        i += 1

        # Wait for the 'q' key to be pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    input_pattern = './frames/Frame_%03d.png'
    output_video_file = 'output_video.mp4'
    fps = 30
    track_video_file = 'tracking.mp4'
    # create_video(frames_patten=input_pattern, video_file = track_video_file, framerate=fps)
    ffmpeg.input( './frames/Frame_%03d.png', framerate=30).output('output_video.mp4',codec='libx264', pix_fmt='yuv420p').run()


if __name__ == "__main__":
    main()