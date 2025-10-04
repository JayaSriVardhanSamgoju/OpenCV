import cv2
import numpy as np

circles = np.zeros((4, 2), np.int32)
counter = 0


scale_display = 0.5  

def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        orig_x = int(x / scale_display)
        orig_y = int(y / scale_display)
        circles[counter] = (orig_x, orig_y)
        counter += 1
        print(f"Point {counter}: {orig_x}, {orig_y}")

# Load the image
img = cv2.imread('DJ.jpg')
if img is None:
    print("Error: Image not found")
    exit()

# Resize image for display
img_display = cv2.resize(img, None, fx=scale_display, fy=scale_display)

cv2.namedWindow('Original Image')
cv2.setMouseCallback('Original Image', mousePoints)

while True:
    img_temp = img_display.copy()
    
    # Draw points on resized display image
    for i in range(counter):
        # Scale points for display
        display_x = int(circles[i][0] * scale_display)
        display_y = int(circles[i][1] * scale_display)
        cv2.circle(img_temp, (display_x, display_y), 5, (0,255,0), cv2.FILLED)
    
    cv2.imshow('Original Image', img_temp)
    
    # Apply perspective transform after 4 points
    if counter == 4:
        width, height = 350, 350
        pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        
        # Resize the warped image
        imgOutput_resized = cv2.resize(imgOutput, None, fx=1.5, fy=1.5)
        cv2.imshow('Warped Image Resized', imgOutput_resized)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
