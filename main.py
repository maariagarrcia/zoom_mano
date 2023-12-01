import cv2
import mediapipe as mp
# import time


def landMarkToCoodinates(tip, dim) -> list:
    return (int(tip.x*dim[0]), int(tip.y*dim[1]))


def scaleDimension(dim, scale) -> list:
    return (int(dim[0]*scale), int(dim[1]*scale))


def distance(p1, p2): # distancia entre dos puntos q se usara 
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return abs(dis)


def scalingProcess(hand, inc):
    tip_4 = landMarkToCoodinates(hand.landmark[4], originalDim)
    tip_8 = landMarkToCoodinates(hand.landmark[8], originalDim)
    internal_6 = landMarkToCoodinates(hand.landmark[6], originalDim)
    internal_3 = landMarkToCoodinates(hand.landmark[3], originalDim)

    dis1 = distance(tip_4, tip_8)
    dis2 = distance(internal_3, internal_6)

    ratio = dis1/dis2
    if ratio > 1.22:
        return inc, [tip_4, tip_8]
    elif ratio < 0.67:
        return -inc, [tip_4, tip_8]

    return 0, ()


def mpHandsInit():
    mpHands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.95)
    mpDraw = mp.solutions.drawing_utils

    return mpHands, mpDraw


def frontalCameraInit():
    camera = cv2.VideoCapture(0)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return camera, (width, height)


def scaleInit(originalDim, scale):
    scaledDim = scaleDimension(originalDim, scale)

    return scale, scaledDim


def showHands(frame, mpDraw, multiHandsLandMarks, activePoints=[], showHands=False):
    if multiHandsLandMarks and showHands:
        for handLandmarks in multiHandsLandMarks:
            mpDraw.draw_landmarks(
                frame, handLandmarks, mp.solutions.hands.HAND_CONNECTIONS)


def showScaleActivePoints(frame, activePoints, scaleInc, drawSymbol=True, drawCircle=False):
    symbol = " -"
    if scaleInc > 0:
        symbol = " +"

    for point in activePoints:
        if drawSymbol:
            cv2.putText(
                frame, symbol, point, cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 255, 0), 4, cv2.LINE_AA, True)

        if drawCircle:
            cv2.circle(frame, point, 12, (0, 255, 0), 3)


def processHands(frame):
    scaleActivePoints = []
    scaleInc = 0
    results = mpHands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for idx, handLandmarks in enumerate(results.multi_hand_landmarks):
            handLabel = results.multi_handedness[idx].classification[0].label
            if handLabel.lower() == "right":
                scaleInc, scaleActivePoints = scalingProcess(
                    handLandmarks, 0.007)

    return scaleInc, results.multi_hand_landmarks, scaleActivePoints


#####
#
#  MAIN
#
#####
mpHands, mpDraw = mpHandsInit()
frontalCam, originalDim = frontalCameraInit()
scale, scaledDim = scaleInit(originalDim, 0.5)
lineColor = (0, 255, 0)
multiHandsLandMarks = None

while frontalCam.isOpened() and cv2.waitKey(10) != 113:
    success, frame = frontalCam.read()
    if not success:
        break  # ======================================>

    frame = cv2.flip(frame, flipCode=1)

    scaleInc, multiHandsLandMarks, scaleActivePoints = processHands(frame)

    showHands(frame, mpDraw, multiHandsLandMarks, scaleActivePoints, True)
    showScaleActivePoints(frame, scaleActivePoints, scaleInc)

    scale = min(2, max(0.1, scale + scaleInc))
    scaledDim = scaleDimension(originalDim, scale)
    frame = cv2.resize(frame, scaledDim)

    cv2.imshow("Webcam Pipe", frame)

cv2.destroyAllWindows()
print("· Ending capture")

