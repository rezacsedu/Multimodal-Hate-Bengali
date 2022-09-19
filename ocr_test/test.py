import cv2
import pytesseract
from PIL import Image

file_path = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/images/3.jpeg"
filename = "/Users/Sumon/Desktop/Multimodal-Hate-Speech-Bengali/"

img_cv = cv2.imread(file_path)

# By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
# we need to convert from BGR to RGB format/mode:
#img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
img_rgb= cv2.bilateralFilter(img_cv,5, 55,60)

print(pytesseract.image_to_string(file_path,lang='ben'))

cv2.imwrite("result.jpeg",img_rgb)
file_text = pytesseract.image_to_string(img_rgb,lang='ben')

text_file_name = filename+"scanned.txt" 
with open(text_file_name, "a") as f:
    f.write(file_text + "\n")
    