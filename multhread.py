
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

from timeit import repeat
from multiprocessing import Pool
import threading







backSub = cv.createBackgroundSubtractorMOG2()
# list to store clicked coordinates
coords = []
def click_event(event, x, y, flags, params):
    
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        
        
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        coords.append((x,y))
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame1, str(x) + ',' +
                   str(y), (x,y), font,
                   1, (255, 0, 0), 2)
        cv.imshow('image', frame1)
    
# checking for right mouse clicks
    if event==cv.EVENT_RBUTTONDOWN:
        
        
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        coords.append((x,y))
        
        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = frame[y, x, 0]
        g = frame[y, x, 1]
        r = frame[y, x, 2]
        cv.putText(frame1, str(b) + ',' + str(g) + ',' + str(r), (x,y), font, 1, (255, 255, 0), 2)
        cv.imshow('image', frame1)
        
        
# return coordinates of frame
def get_coords(frame):
    cv.imshow('image', frame)
    
    # setting mouse hadler for the image
    # and calling the click_event() function
    cv.setMouseCallback('image', click_event)
    
    # wait for a key to be pressed to exit
    cv.waitKey(0)
    #cv.destroyAllWindows()
    
    pts = np.asarray(coords)
    rect = cv.boundingRect(pts)
    
    
    return rect, pts

# cut the given frame and rect with np array of coords
def cut_image(frame, rect, pts):
    x,y,w,h = rect
    croped = frame[y:y+h, x:x+w].copy()
    
    ## (2) make mask
    pts = pts - pts.min(axis=0)
        
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
        
    ## (3) do bit-op
    dst = cv.bitwise_and(croped, croped, mask=mask)
    
    return dst


def process(frames):
    filee = open(r"labels.txt", "a")
    for frame in frames:
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
        fg = backSub.apply(blurred)
        output = cv.connectedComponentsWithStats(fg, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output
        for i in range(0, numLabels):
           
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]

            label_text = "person" ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            if area > 400 and area < 1000:
                filee.write(label_text)
                
if __name__=="__main__":
    capture = cv.VideoCapture(cv.samples.findFileOrKeep("right_sample2.mov"))
    coords = [(931,318),( 0,366), (223,974), (1905,577)]
    points = np.asarray(coords)
    shape = cv.boundingRect(points)

    if not capture.isOpened():
        print('Unable to open: ')
        exit(0)
    
    # store all frames in a list
    frames = []
    print('reading frames...')
    start_read = time.time()
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        image = cut_image(frame, shape, points)
        frames.append(image)
    end_read = time.time()
    
    print('processing frames...')

    # make 5 chunks
    chunks = [frames[i::5] for i in range(5)]
    tasks = []
    start_process = time.time()
    for chunk in chunks:
        tasks.append(threading.Thread(target=process, args=(chunk,)))
        tasks[-1].start()

    for task in tasks:
        task.join()

    end_process = time.time()

   
    print('read time: ', end_read-start_read)
    print('process time: ', end_process-start_process)
   
     
