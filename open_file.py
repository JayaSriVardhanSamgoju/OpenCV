import cv2
import numpy as np

'''def empty(a):
    pass

# Load image
img = cv2.imread("Dj.jpg")
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

# Create trackbars for HSV values
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get values from trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # Define lower and upper HSV range
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask
    mask = cv2.inRange(imgHSV, lower, upper)

    # Apply mask to get the detected color
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show windows
    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''


'''import cv2

def getContours(img):
    # Find contours
    contours, hierarchy = cv2.findContours(
        img,                          # binary image (edges/mask)
        cv2.RETR_EXTERNAL,            # retrieve only external contours
        cv2.CHAIN_APPROX_SIMPLE       # contour approximation method
    )
    
    for cnt in contours:
        area = cv2.contourArea(cnt)   # calculate area
        if area > 500:                # filter out small noise
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)  # draw contour

            peri = cv2.arcLength(cnt, True)  # perimeter
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # approx polygon
            print(len(approx))  # number of corner points
            x, y, w, h = cv2.boundingRect(approx)  # bounding box
            cv2.rectangle(imgResult, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle
            cv2.putText(imgResult, "Points: " + str(len(approx)), (x + w + 20, y + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)  # put text
            cv2.putText(imgResult, "Area: " + str(int(area)), (x + w + 20, y + 45),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)  # put text
    return imgResult
img = cv2.imread("Dj.jpg")
imgResult = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7),
                          1)  # (7,7) kernel size, sigmaX=1 for Gaussian blur
imgCanny = cv2.Canny(imgBlur, 50, 50)  # Canny edge detection
imgResult = getContours(imgCanny)   # find and draw contours 
cv2.imshow("Original", img)
cv2.imshow("Gray", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Contours", imgResult)
cv2.waitKey(0)

'''

# Load image
'''img = cv2.imread('IMG20230731112241.jpg')
if img is None:
    raise FileNotFoundError("Image not found!")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Convert to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(imgGray, 1.1, 4)

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)

# 🔹 Resize image for display (50% smaller)
imgSmall = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Show result
cv2.imshow('Detected Faces', imgSmall)'''



for i in range(5):
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera Index {i}", frame)
            cv2.waitKey(1000)   # show each for 1 second
        cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()

