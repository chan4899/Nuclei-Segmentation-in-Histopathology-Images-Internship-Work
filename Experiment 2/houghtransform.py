import cv2
import numpy as np

img = cv2.imread('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/test_data/results3/pred_584.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# minLineLength = 400
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# cv2.imwrite('houghlines5.jpg',img)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# dilated = cv2.dilate(img, kernel)
blur = cv2.bilateralFilter(img,9,75,75)
cv2.imwrite('bilateral_filter.jpg',blur)