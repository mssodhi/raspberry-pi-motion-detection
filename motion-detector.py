# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import threading
import numpy as np

def deleteFrame():
    global firstFrame
    firstFrame = None
    # call deleteFrame() every 30 seconds -> .5 to delte previous frame.
    threading.Timer(.5, deleteFrame).start()

deleteFrame()

# set the camera
camera = cv2.VideoCapture(0)

# initialize the first frame in the video stream
global firstFrame
firstFrame = None

# loop over the frames of the video
while 1:
    
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()
    text = "False"
        
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
        
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
        
    # compute the absolute difference between the current frame and
    # first frame
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_3_frame = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2RGB)
    frameDelta = cv2.absdiff(firstFrame, gray)
    frame_3_frame = cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2RGB)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    # thresh_3_frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    adaptive_thresh_3 = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < 500:
            continue
                
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "True"
                    
    # draw the text and timestamp on the frame
    cv2.putText(frame, "Motion Detection: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                                         
    # show the frame and record if the user presses a key
    vstack1 = np.vstack((frame, frame_3_frame))
    vstack2 = np.vstack((gray_3_frame, adaptive_thresh_3))
    numpy_horizontal = np.hstack((vstack1, vstack2))
    cv2.imshow("Monitor", numpy_horizontal)
    # cv2.imshow("Security Feed", frame)
    # cv2.imshow("Gray", grayFrame)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Thresh", adaptive_thresh)
    # cv2.imshow("Frame Delta", frameDelta)

    key = cv2.waitKey(1) & 0xFF
                                                         
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()