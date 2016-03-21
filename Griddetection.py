import cv2
import numpy as np
# from matplotlib import pyplot as plt

image = cv2.imread('sudoku.qy.jpg',0)
h, w = image.shape
factor = 1.0
maxwh = 600
if h > w and h > maxwh:
    factor = 1.0*maxwh/h
elif h <= w and w > maxwh:
    factor = 1.0*maxwh/w
image = cv2.resize(image, (0,0), fx=factor, fy=factor) 

# outerBox = np.zeros(image.shape, dtype=np.uint8)
image = cv2.GaussianBlur(image, (11,11), 0)
gray_image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,5,2)
# print gray_image.shape, gray_image[0][0]
gray_image = cv2.bitwise_not(gray_image)
kernel = np.matrix([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
gray_image = cv2.dilate(gray_image, kernel)


# int count=0;
#     int max=-1;
# 
#         Point maxPt;
# 
#             for(int y=0;y<outerBox.size().height;y++)
#                 {
#                         uchar *row = outerBox.ptr(y);
#                         for(int x=0;x<outerBox.size().width;x++)
#                         {
#                             if(row[x]>=128)
#                             {
# 
#                                 int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,64));
# 
#                                 if(area>max)
#                                 {
#                                     maxPt = Point(x,y);
#                                     max = area;
#                                     }
#                                 }
#                             }
# 
#                         }

maxe = -1
h, w = gray_image.shape
mask = np.zeros((h+2, w+2), np.uint8)
for y in xrange(h):
    for x in xrange(w):
        if gray_image[y][x] >= 128:
            # area = cv2.floodFill(gray_image, mask, (x,y), (0,0,64))
            area = cv2.floodFill(gray_image, mask, (x,y), 64)
            if area > maxe:
                maxe = area
                maxPt = (x,y)
                #print maxe, x, y 

# print maxPt
mask = np.zeros((h+2, w+2), np.uint8)
# cv2.floodFill(gray_image, mask, maxPt, (255,255,255))
cv2.floodFill(gray_image, mask, maxPt, 255)

# if(row[x]==64 && x!=maxPt.x && y!=maxPt.y)
#             {
#                                     int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
#                                                 }

mask = np.zeros((h+2, w+2), np.uint8)
for y in xrange(h):
    for x in xrange(w):
        if gray_image[y][x] == 64 and (x, y) != maxPt: #and x != maxPt.x and y != maxPt.y:
            # area = cv2.floodFill(gray_image, mask, (x,y), (0,0,0))
            area = cv2.floodFill(gray_image, mask, (x,y), 0)

cv2.erode(gray_image, kernel)

lines = cv2.HoughLines(gray_image,1,np.pi/180,120)
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
        # print (x1,y1),(x2,y2)
        cv2.line(gray_image, (x1,y1), (x2,y2), 255, 2)

cv2.imshow("thresholded", gray_image)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()        # Closes displayed windows
# plt.imshow(gray_image)
# plt.show()


