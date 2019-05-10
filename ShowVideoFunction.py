# coding: utf-8
import numpy as np
import cv2
import os


def ShowVideo(dist1,dist2,delaytime = 100):

    cap1 = cv2.VideoCapture(dist1)
    cap2 = cv2.VideoCapture(dist2)
    if cap1.isOpened() != True & cap2.isOpened() != True:
        os._exit(-1)
     
    width1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height1 = (int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    width2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height2 = (int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    #this print is used to debug the play speed
    #print ('delaytimeis:',delaytime)
    
    while True:
        # capture one by one
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if ret1 != True and ret2 != True:
            break
     
        
        # Test for make the video gray
        #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        #resize the window size 'interpolation'can fill the space, make sure the video correct.
        gray1 = cv2.resize(frame1, (int(width1), int(height1)), interpolation=cv2.INTER_CUBIC)
        gray2 = cv2.resize(frame2, (int(width2 / 2), int(height2 / 2)), interpolation=cv2.INTER_CUBIC)

     
        cv2.imshow(dist1, gray1)
        cv2.imshow(dist2, gray2)
        #the play speed was litel bit fast so add thie for slowdown,and make it could be adjust from outside
        #put this before the Q key and space key because need these two keep have good responsbility
        cv2.waitKey(delaytime)
     
        # use space key to pause
        if (cv2.waitKey(1) & 0xFF) == ord(' '):
            cv2.waitKey(0)

        # use Q key to exit
        if (cv2.waitKey(10) & 0xFF) == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def main():
    print('this message is from main function')
    dist1 = './challenge_video_with_debug_window.mp4'
    dist2 = './challenge_video.mp4'
    delaytime = 100

    #ShowVideo(dist1,dist2,delaytime)
    #ShowVideo(dist1,dist2)
    ShowVideo('./output_video/temp/project_video.mp4', './test_video/project_video.mp4', delaytime = delaytime)
    #ShowVideo('./output_video/temp/challenge_video.mp4', './test_video/challenge_video.mp4', delaytime = 15)
    #ShowVideo('./output_video/temp/harder_challenge_video.mp4', './test_video/harder_challenge_video.mp4', delaytime = 15)

if __name__ == '__main__':
    main()
    # print(__name__)









