import cv2
import numpy as np
import time

mouseX = 0
mouseY = 0
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))
       global mouseX
       global mouseY
       mouseX = x
       mouseY = y

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
#cv2.namedWindow("test2")

cv2.setMouseCallback('test', onMouse)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 24)
img_counter = 1
start = time.time()
begin = False

blk_size = 11
blur = 1

def on_trackbar(val):
    global blk_size
    if (val % 2 == 0):
        val = val + 1
    blk_size = val

def on_blurTrack(val):
    global blur
    if (val % 2 == 0):
        val = val + 1
    blur = val

cv2.createTrackbar("block size", "test", 0, 2000, on_trackbar)
cv2.createTrackbar("blur", "test", 0, 25, on_blurTrack)
while True:
    ret, frame = cam.read()
    # cv2.circle(frame, (85, 0), 3, (255, 0, 0), 2)
    # cv2.circle(frame, (310, 0), 3, (255, 0, 0), 2)
    # cv2.circle(frame, (85, 256), 3, (255, 0, 0), 2)
    # cv2.circle(frame, (310, 256), 3, (255, 0, 0), 2)
    low_H = 0
    low_S = 108
    low_V = 0
    high_H = 180
    high_S = 251
    high_V = 107
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, imgray = cv2.split(frame_HSV)
    frame = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
  
    #imgray = cv2.cvtColor(frame, cv2.COLOR_HSV2GRAY)


    thresh = cv2.adaptiveThreshold(imgray , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, 1)
    blurred = cv2.GaussianBlur(imgray, (0, 0), 1)
    #thresh = cv2.adaptiveThreshold(blurred , 255 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, 1)

    # frame = cv2.erode(thresh, None, iterations = 1)
    # frame = cv2.dilate(thresh, None, iterations = 1)

    #blurred = cv2.blur(thresh, (5, 5))
    frame = thresh
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.rectangle(frame, (260, 0), (516, 260), (255, 0, 0), 1)
    

    
    # for contour in contours:
    #     if cv2.contourArea(contour) > 700:
    #         x,y,w,h = cv2.boundingRect(contour) # offsets - with this you get 'mask'
    #         cv2.rectangle(frame ,(x,y),(x+w,y+h),(255,255,0),2)
    #         mean = np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(np.uint8)

    #         if (mouseX > x) and (mouseX < x+w) and (mouseY > y) and (mouseY < y+h):
    #             print(mean)
    #             mouseX = mouseY = 0
            # print('Average color (BGR): ',np.array(cv2.mean(frame[y:y+h,x:x+w])).astype(np.uint8))
           

    #frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    # if begin:
    #     if time.time() - start >= 0.1:
    #         img_name = "Illinois{}.png".format(img_counter)
    #         cv2.imwrite(img_name, frame)
    #         print("{} written!".format(img_name))
    #         img_counter += 1
    #         start = time.time()

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # print("Capture start...")
        img_name = "Kentucky{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        # begin = True
 

cam.release()

cv2.destroyAllWindows()