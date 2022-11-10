# imports for openCV, MediaPipe, and numpy
import cv2
import mediapipe as medPipe
import numpy as np

# two libraries from mediapipe to recognize poses
drawings = medPipe.solutions.drawing_utils
poseSolutions = medPipe.solutions.pose

# CAPTURE THE VIDEO
cap = cv2.VideoCapture(0)

# INIT VARIABLE FOR REPS AND MOTION
reps = 0
motion = None

# calculate angle of each position between joints


def calcAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rads = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = np.abs(rads*180.0/np.pi)

    if ang > 180.0:
        ang = 360-ang

    return ang


# creating mediapipe instance with a 70 percent confidence accuracy
with poseSolutions.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as poseTrack:
    while cap.isOpened():
        ret, frame = cap.read()

        # change image to rgb to allow mediapipe to process images
        imageColoring = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imageColoring.flags.writeable = False

        # detect the image for joints
        detections = poseTrack.process(imageColoring)

        # change image back to brg to allow openCV to process image
        imageColoring.flags.writeable = True
        imageColoring = cv2.cvtColor(imageColoring, cv2.COLOR_RGB2BGR)

        # Get the landmarks of all body positions
        try:
            bodyPositions = detections.pose_landmarks

            # store all coordinates of all the landmark poses
            landmarkList = []

            # if body is present in camera
            if bodyPositions:
                drawings.draw_landmarks(
                    imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS)
                # iterate thru all of the landmarks to store and compare later
                for id, landmark in enumerate(detections.pose_landmarks.landmark):
                    print(id)
                    height, width, _ = imageColoring.shape
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    landmarkList.append([id, x, y])
            # check if body was present
            if len(landmarkList) != 0:
                # logic to check if joint positions of left/right shoulder and left/right elbow change
                if ((landmarkList[12][2] - landmarkList[14][2]) >= 15 and
                        (landmarkList[11][2] - landmarkList[13][2]) >= 15):
                    motion = "  down"
                if ((landmarkList[12][2] - landmarkList[14][2]) <= 5 and
                        (landmarkList[11][2] - landmarkList[13][2]) <= 5) and motion == "  down":
                    motion = "  up"
                    reps += 1
                    print(reps)

        except:
            pass

        # VARS FOR TEXT FIELD
        image = imageColoring
        font = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 2.5
        color = (255, 255, 0)
        thickness = 2

        # COUNTER TEXT
        cv2.putText(image, str(reps), (10, 60), font,
                    fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(image, motion, (60, 60), font,
                    fontScale, color, thickness, cv2.LINE_AA)

        # Render the landmarks and dictate their color and thickens
        drawings.draw_landmarks(imageColoring, detections.pose_landmarks, poseSolutions.POSE_CONNECTIONS, drawings.DrawingSpec(
            color=(255, 0, 255), thickness=2, circle_radius=2), drawings.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2))

        cv2.imshow('Image Feed', imageColoring)

        # Exit by pressing 'X'
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()
