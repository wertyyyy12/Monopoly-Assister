import cv2
import numpy as np  
import time

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
cv2.namedWindow("otherImage")
#cv22.namedWindow("test2")
# cam.set(cv22.CAP_PROP_FRAME_WIDTH, 64)
# cam.set(cv22.CAP_PROP_FRAME_HEIGHT, 24)
img_counter = 1
begin = False
start = time.time()
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, imgray = cv.split(frame_HSV)
    max_value = 255
    max_value_H = 360//2
    low_H = 0
    low_S = 72
    low_V = 0
    high_H = 95
    high_S = max_value
    high_V = max_value
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # blurred = cv.GaussianBlur(frame_threshold, (0, 0), 1)
    blurred = cv2.medianBlur(frame_threshold, 5)
    blurred = cv2.medianBlur(blurred, 5)
    
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # blurred = 

    for c in contours:
        # if cv.contourArea(c) > :
        maxC = max(contours, key = cv2.contourArea)    
        # x,y,w,h = cv.boundingRect(c) # offsets - with this you get 'mask'
        rect = cv2.minAreaRect(maxC)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv.rectangle(frame ,(x,y),(x+w,y+h),(255,255,0),2)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(frame, M, (width, height))
        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        warped = cv2.resize(warped, (64, 36))
        cv2.imshow("otherImage", warped)

    if begin:
        if time.time() - start >= 0.1:
            img_name = "Indiana{}.png".format(img_counter)
            cv2.imwrite(img_name, warped)
            print("{} written!".format(img_name))
            img_counter += 1
            start = time.time()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        print("Capture start...")
        # img_name = "Kentucky{}.png".format(img_counter)
        # cv2.imwrite(img_name, warped)
        # print("{} written!".format(img_name))
        # img_counter += 1
        begin = True
 

cam.release()

cv2.destroyAllWindows()