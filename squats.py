# imports for openCV, MediaPipe, and numpy
import cv2
import mediapipe as medPipe
import numpy as np

# two libraries from mediapipe to recognize poses
drawings = medPipe.solutions.drawing_utils
poseSolutions = medPipe.solutions.pose

# CAPTURE THE VIDEO
cap = cv2.VideoCapture(0)

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

        # Get the landmarks of the shoulders,hip, knee, and ankle to calculate the angle
        try:
            bodyPositions = detections.pose_landmarks.landmark

            # GET SQUAT COORDINATES
            # only need one side so we are choosing right landmarks
            R_hip = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_HIP.value].x,
                     bodyPositions[poseSolutions.PoseLandmark.RIGHT_HIP.value].y]
            R_knee = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_KNEE.value].x,
                      bodyPositions[poseSolutions.PoseLandmark.RIGHT_KNEE.value].y]
            R_ankle = [bodyPositions[poseSolutions.PoseLandmark.RIGHT_ANKLE.value].x,
                       bodyPositions[poseSolutions.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate the knee-joint angle and the hip-joint angle
            kneeAngle = calcAngle(R_hip, R_knee, R_ankle)
            kneeAngle = round(kneeAngle, 2)

            # if standing up
            if kneeAngle > 150:
                motion = "  up"
            # if doing squat motion down
            if kneeAngle <= 120 and motion == "  up":
                print(kneeAngle)
                motion = "  down"
                reps += 1

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
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cap.release()
