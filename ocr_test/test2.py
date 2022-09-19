import cv2
import pytesseract
from PIL import Image

file_path = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/images/3.jpeg"
filename = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/"


image = cv2.imread(file_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Morph open to remove noise and invert image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening

# Perform text extraction
data = pytesseract.image_to_string(invert, lang='ben', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')#config='--psm 6')
print(data)

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('invert', invert)
cv2.waitKey()