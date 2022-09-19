# importing all required libraries
import os
import traceback
from PIL import Image

# importing libraries for computer vision
import numpy as np
import cv2 
import imutils
from imutils import contours
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
tesseract_cmd = '/Users/Sumon/opt/anaconda3/envs/nlp/bin/pytesseract'

# importing libraries to read text from image
from PIL import Image
import pytesseract
filename = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/test.jpeg"
file_path = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/images/8.jpeg"
img1 = cv2.imread(file_path)
img1 = cv2.resize(img1, (600,800))
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


ret,thresh1 = cv2.threshold(img,127,255,0)
#cons, contours, sd = cv2.findContours(thresh1, thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cnt = contours[1019]
print(len(contours))
cv2.drawContours(img1, contours, -1, (0,255,0), 3)
# cv2.imshow('Contours', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# x,y,w,h = cv2.boundingRect(cnt)
# cv2.rectangle(img1,(x,y),(x+w,y+h),(0,255,0),2)
a,b,max_w,max_h = cv2.boundingRect(contours[359])
max_x = 0
max_y = 0

for i in range(0,len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    if max_h*max_w < w*h:
        max_h = h
        max_w = w
        max_a = x
        max_b = y

_,_,second_max_w,second_max_h = cv2.boundingRect(contours[359])

second_max_x = 0
second_max_y = 0
for i in range(0,len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    if second_max_h*second_max_w < w*h and w != max_w and h != max_h :
        second_max_h = h
        second_max_w = w
        second_max_a = x
        second_max_b = y

cv2.rectangle(img1,(max_a,max_b),(max_a+max_w,max_b+max_h),(0,255,255),2)
cv2.rectangle(img1,(second_max_a,second_max_b),(second_max_a+second_max_w,second_max_b+second_max_h),(0,255,0),2)

card = img1[max_b:max_b+max_h,max_a:max_a+max_w]
name_area = thresh1[second_max_b:second_max_b+second_max_h,second_max_a:second_max_a+second_max_w]

print("xcoord {} ycoord {} height {} width {}".format(max_a,max_b,max_h,max_w))

cv2.imwrite(filename,cv2.bitwise_not(name_area))
r = Image.open(filename)
r.load()
text = pytesseract.image_to_string(r)
#os.remove('name.png')
print(text)


