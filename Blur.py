import cv2
import numpy as np

img  = cv2.imread('shapes.jpg')
blur = cv2.blur(img,(10,10))
gaussian = cv2.GaussianBlur(img,(5,5),0) 
median = cv2.medianBlur(img, 7)
cv2.imshow('Average image',blur)
cv2.imshow('gaussian image',gaussian)
cv2.imshow('median image',median)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

