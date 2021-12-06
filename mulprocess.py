
import cv2 as cv
import numpy as np
import time

from timeit import repeat
from multiprocessing import Pool
from collections import deque
from multiprocessing.pool import ThreadPool

backSub = cv.createBackgroundSubtractorMOG2()



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
        # hardcoded points for boundary
        coords = [(931,318),( 0,366), (223,974), (1905,577)]
        points = np.asarray(coords)
        shape = cv.boundingRect(points)
        frame = cut_image(frames, shape, points)

    
        # background subtraction
        blurred = cv.GaussianBlur(frame, (5, 5), 0)
        fg = backSub.apply(blurred)
        # connected components
        output = cv.connectedComponentsWithStats(fg, 4, cv.CV_32S)
        (numLabels, labels, stats, centroids) = output

        for i in range(0, numLabels):
           
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area = stats[i, cv.CC_STAT_AREA]

            label_text = "person" ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
            # threshold
            if area > 400 and area < 1000:
                filee.write(label_text)
                
if __name__=="__main__":
    
    
    cap = cv.VideoCapture('right_sample2.mov')
    thread_num = cv.getNumberOfCPUs()
    print('Running on ', thread_num, ' CPUs.')
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()
    start = time.time()
    while True:
        # Consume the queue.
        while len(pending_task) > 0 and pending_task[0].ready():
            res = pending_task.popleft().get()

        # Populate the queue.
        if len(pending_task) < thread_num:
            frame_got, frame = cap.read()
            if frame_got:
                task = pool.apply_async(process, (frame.copy(),))
                pending_task.append(task)


        if not frame_got:
            break
    end_read = time.time()
    
    print('Time taken: ', end_read-start)
    #print('write time: ', end_write-start_write)

