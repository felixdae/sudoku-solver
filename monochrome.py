# import cv2
# image = cv2.imread('sudoku.png')
# print len(image), len(image[0]), image[0][0];
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# print len(gray_image),gray_image[0]
# cv2.imwrite('gray_image.png',gray_image)
# cv2.imshow('color_image',image)
# cv2.imshow('gray_image',gray_image) 
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()        # Closes displayed windows

#End of Code


import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

# image = cv2.imread('sudoku.png',0)
# image = cv2.imread('sudoku.wiki.png',0)
image = cv2.imread('sudoku.qy.jpg',0)
print type(image)
image = cv2.resize(image, (0,0), fx=0.3, fy=0.3) 
# print len(image), len(image[0])
# image = cv2.medianBlur(image,5)

# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
gray_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# titles = ['Original Image', 'Global Thresholding (v = 127)',
#            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# images = [th2, th3]

# for i in xrange(2):
#     plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
cv2.imwrite('gray_image.png',gray_image)
# plt.imshow(gray_image, 'gray')
# plt.show()
# sys.exit(1)
# print len(gray_image), len(gray_image[0])
# cv2.imshow('color_image',image)
# cv2.imshow('gray_image',gray_image) 
# cv2.waitKey(0)                 # Waits forever for user to press any key
# cv2.destroyAllWindows()        # Closes displayed windows

# print gray_image.shape

edges = cv2.Canny(gray_image,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,120)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# print lines, len(lines),len(lines[0])
for line in lines:
    # for x1,y1,x2,y2 in line:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        print (x1,y1),(x2,y2)

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('houghlines3.jpg',image)
plt.imshow(image)
plt.show()
