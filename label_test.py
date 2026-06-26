import cv2 
import os

image_path = os.path.join(os.getcwd(),'data','flickr-image-dataset','versions','1','flickr30k_images','flickr30k_images','18638962.jpg')
label_path = os.path.join(os.getcwd(),'data','labels','18638962.txt')

img = cv2.imread(image_path)
H, W = img.shape[:2]
for line in open(label_path,'r'):
    cls, xc, yc, w, h = map(float,line.split())
    x1 = int((xc - w/2) * W)
    y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W)
    y2 = int((yc + h/2) * H)
    cv2.rectangle(img,(x1,y1),(x2,y2),(255, 0, 0), 2)    
cv2.imwrite("check.jpg", img)