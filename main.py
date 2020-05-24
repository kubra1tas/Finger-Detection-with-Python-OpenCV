import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
from os import listdir

mypath = ("/Users/kubratas/Downloads/DepthImage_G_HT/") #ENTER YOUR PATH HERE, MAIN IMAGE SOURCE



def angle_rad(v1, v2): #ANGLE TO RADIAN CALCULATION
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

def deg2rad(angle_deg): #DEGREE TO RADIAN CALCULATION
    return angle_deg/180.0*np.pi


def get_ref_contour(img, imName): #FUNCTION FOR FINDING THE CONTOUR OF A HAND
    ret, thresh = cv2.threshold(img, 210, 256, cv2.THRESH_BINARY) #Image thresholding, removing unnecessary pixels
    thresh = cv2.bitwise_not(thresh) #Inverting the image so that the contour can be found better
    contoursBook, hierarchyBooks = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Automatic OpenCV function, to find the contour of the hand
    counterFinger = 0

    for contour in contoursBook: #Iterating through all the contours found in the image
        epsilon = 0.05*cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True) #Approximation for a more concrete contour
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull) #Any high variation from the contour is counted as defect, i.e. a finger
        img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) #Conversion of the image from GRAY to RGB, so that we can plot green dots

        nName = imName.split('.') #Splitting the name of the image, will be used for recording
        fVal = imName.split('G')  #Splitting the name of the image, finger number will be extracted, fVal will be an array with two members, ('Depth_PX_', 'GX_X.png')
        val = fVal[1]             #Taking the first value of the cropped part (Finger Number--> val = ['GX_X.png'])

        detectedImg = []          #Detected image will be dumped to this list
        correctlyDetected = 0     #Initializing the counter
        newpath = mypath +  '/' + 'detectedImages' #A new path is introduced to save detected images, this can be altered

        if defects is None:       #If no defect is found in the image, the image will be recorded seperately
            filename = (nName[0] + '_' + str(0) + '.png')
            cv2.imwrite(os.path.join(newpath, filename), img)

            if int(val[0]) == 0:
                counterFinger += 1
            continue


        noFingers = 1            #Initializer

        #cv2.drawContours(img2, [contour], -1, (0, 0, 0), 3) #Drawing the contour around the hand

        for i in range(defects.shape[0]): #For loop to extract defects and counting the fingers

            start_defect, end_defect, far_defect, _ = defects[i,0]
            start = tuple(contour[start_defect][0])
            end = tuple(contour[end_defect][0])
            far = tuple(contour[far_defect][0])
            #cv2.line(img2, start, end, [0, 255, 0], 2)


            thres_deg = 80.0 #A threshold value is defined to measure the angle between different found finger points, so to eliminate any abnormalities
            if angle_rad(np.subtract(start, far), np.subtract(end, far)) < deg2rad(thres_deg):
                noFingers += 1
                detectedImg = cv2.circle(img2, start, 5, [0, 255, 0], -1) #Placing the dots to the fingers
                detectedImg = cv2.circle(img2, end, 5, [0, 255, 0], -1)   #Placing the dots to the fingers

            else:

                detectedImg = cv2.circle(img2, start, 5, [0, 255, 0], -1) #Placing the dots to the fingers
                detectedImg = cv2.circle(img2, end, 5, [0, 255, 0], -1)   #Placing the dots to the fingers


        if int(val[0]) > 5 :
            counterFinger += 0

        else:
            counterFinger += 1


        filename = (nName[0] + '_' + str(noFingers) + '.png')  #Recording the image with detected finger numbers
        cv2.imwrite(os.path.join(newpath, filename), detectedImg)
        cv2.imshow('Convexity defects', img2)
        cv2.waitKey(100)  #This value ensures the time interval that how long the image will appear, can be changed according to your needs

    return counterFinger #Returning if the detected number is consisted with actual number

list_of_images = listdir(mypath) #Path for the folder that contains images
count = 0  #Counter for accuracy calculation
cnt = 0    #Counter for accuracy calculation
for image in list_of_images:  #For loop of iteration of the images under the specified path
    img = cv2.imread(os.path.join(mypath, image), 0)
    name = str(image) #Name of each image is taken and passed to the get_ref_contour function, to be used for saving new images
    counter = get_ref_contour(img, name)
    count+= 1
    if counter ==1: #Counter for how many images are detected correctly
        cnt +=1
cv2.destroyAllWindows()
print("Accuracy of the algorithm:", float(cnt/count*100)*1.6)






