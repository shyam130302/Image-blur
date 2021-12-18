
import cv2
import numpy as np

  
def detect_shapes(img):
    detected_shapes = []    
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    binary = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)  
    contures,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    for c in contures:        
        M = cv2.moments(c)
        x = -1
        y = -1
        if (M['m00']!=0):
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])        

        approx = cv2.approxPolyDP(c,0.02 * cv2.arcLength(c,True),True)
        if len(approx) == 3:
            shape = 'Triangle'
        elif len(approx) == 4:
            (k,l, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle" 
        elif len(approx) == 5:
            shape = 'Pentagon'
        else:
            shape = 'Circle'        

      
        shapeList = [shape,(x,y)]
        detected_shapes.append(shapeList)    
    return detected_shapes

def get_labeled_image(img, detected_shapes):
    for detected in detected_shapes:
        shape = detected[0]
        coordinates = detected[1]
        cv2.putText(img,str(shape),coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img

if __name__ == '__main__':   

    img_file_path = 'shapes.png'   
    img = cv2.imread(img_file_path)    
    print('\n============================================')   
    detected_shapes = detect_shapes(img)
    print(detected_shapes)    # display image with labeled shapes
    img = get_labeled_image(img, detected_shapes)
    cv2.imshow("labeled_image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
