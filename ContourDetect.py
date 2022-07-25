import cv2

video = cv2.VideoCapture('video2.mp4')

# Opencv Background subtractor for k-Nearest Neighbors
# Parameters: history = 100, dist2Threshold = 500, bool detectShadows = False
bskNN = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=500)

vehicles = 0
validVehicles = []

while video.isOpened():
    # read a frame from the video
    ret, frame = video.read()
    # height by width - Region is the area we are working with
    region = frame[350:1080, 205:1500]  # Area coordinates: (205,350), (1500,1080)
    # frame = cv2.rectangle(frame, (205,350), (1500,1080), (0, 0, 255), 2)

    # cv2.imshow('Region', region)
    # cv2.imshow('FrameReg', cv2.resize(frame, (1000, 546)))
    cv2.line(frame, (150, 900), (1600, 900), (0, 0, 255), 2)  # RED Line
    # Applying the background subtractor kNN to the frame
    mask = bskNN.apply(region)
    # cv2.imshow('Mask1', cv2.resize(mask, (1000, 546)))
    # Using threshold of 254, so all pixels >254 will be assigned 255 (white)
    # Performs well to Remove Shadows
    ret1, mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Mask2', cv2.resize(mask, (1000, 546)))

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))

    dilate = cv2.dilate(mask, kernal, iterations=3)
    # cv2.imshow('Dilated Frame', cv2.resize(dilate, (1000, 546)))
    # finding the contours on the Mask frames
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours[0] if len(contours) == 2 else contours[1]
    # print(contours[:5])
    for contour in contours:
        # draw an approximate rectangle around the binary image
        x, y, w, h = cv2.boundingRect(contour)
        # Height and width over a certain pixel size is used for drawing rectangles
        if h > 40 and w > 40:
            # draw rectangles
            cv2.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 3)
            xCenter = int((x + (x + w)) / 2)
            yCenter = int((y + (y + h)) / 2)
            cv2.circle(region, (xCenter, yCenter), 5, (0, 0, 255), 3)
            print(yCenter)
            validVehicles.append((xCenter, yCenter))
            for (X, Y) in validVehicles:
                if 450 < Y < 457:  # For frames
                    vehicles += 1
                    validVehicles.remove((X, Y))
                    # print(vehicles)

    # displaying 2 windows that are resize to be smaller for easier viewing
    cv2.imshow('BS with KNN', cv2.resize(dilate, (1000, 546)))
    cv2.putText(frame, 'Vehicles : {}'.format(vehicles), (800, 50), cv2.FONT_ITALIC, 2, (0, 0, 0), 2)
    cv2.imshow('Frame', cv2.resize(frame, (1000, 546)))
    # cv2.imshow('Frame', frame)

    key = cv2.waitKey(10)
    # hit Esc key on keyboard to exit loop and close video window
    if key == 27:
        break

cv2.destroyAllWindows()
video.release()
