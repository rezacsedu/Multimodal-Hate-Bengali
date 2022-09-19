# importing modules
import cv2
import pytesseract
# reading image using opencv
file_path = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/images/1.jpeg"

image = cv2.imread(file_path)
#converting image into gray scale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# converting it to binary image by Thresholding

# this step is require if you have colored image because if you skip this part

# then tesseract won't able to detect text correctly and this will give incorrect result

threshold_img = cv2.threshold(gray_image, 0, 123, cv2.THRESH_BINARY, cv2.THRESH_OTSU)[1]

# display image

cv2.imshow('threshold_img', threshold_img)

# Maintain output window until user presses a key

cv2.waitKey(0)

# Destroying present windows on screen

cv2.destroyAllWindows()