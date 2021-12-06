

'''

Restructured serial code

'''
import cv2 as cv
import numpy as np
import time
from timeit import repeat
from multiprocessing import Pool
import threading







backSub = cv.createBackgroundSubtractorMOG2()
# list to store clicked coordinates
LABELS = []

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


def process(frame):
    #for frame in frames:
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
                LABELS.append(label_text)
                
if __name__=="__main__":
    capture = cv.VideoCapture(cv.samples.findFileOrKeep("right_sample2.mov"))
    file = open(r"labels.txt", "a")
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

    start_process = time.time()

    for frame in frames:
        process(frame)

    end_process = time.time()
    start_write = time.time()
    for lab in LABELS:
        file.write(lab)
    end_write = time.time()
   
    print('read time: ', end_read-start_read)
    print('process time: ', end_process-start_process)
    print('write time: ', end_write-start_write)


   
     
