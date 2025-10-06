import cv2 
path='C:/Users/Vardhan/Documents/ML Bootcamp/OpenCV/haarcascade_frontalface_default.xml'
cameraNo=0
objectName='Face'
framewidth=640
frameheight=480
cap = cv2.VideoCapture(cameraNo,cv2.CAP_DSHOW)
cap.set(3, framewidth)
cap.set(4, frameheight)

def empty(a):
    pass

#Create a Trackbar
cv2.namedWindow("Results")
cv2.resizeWindow("Results", framewidth, frameheight+100)    
cv2.createTrackbar("Scale", "Results", 400, 1000, empty)
cv2.createTrackbar("Neig", "Results", 8, 20, empty)
cv2.createTrackbar("Min Area", "Results", 0, 100000, empty)
cv2.createTrackbar("Brightness", "Results", 180, 255, empty)

#Load the classifier
cascade = cv2.CascadeClassifier(path)

while True:
    #Set the brightness of the camera
    brightness = cv2.getTrackbarPos("Brightness", "Results")
    cap.set(10, brightness)
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #Get the values from the trackbar
    scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Results") / 1000)
    neig = cv2.getTrackbarPos("Neig", "Results")
    minArea = cv2.getTrackbarPos("Min Area", "Results")

    #Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Detect the object using haarcascade
    objects = cascade.detectMultiScale(imgGray, scaleVal, neig)

    #Draw the rectangle around the object
    for (x, y, w, h) in objects:
        area = w * h
        minArea = cv2.getTrackbarPos("Min Area", "Results") 
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(img, objectName, (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(img, str(area), (x, y + h + 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    #Show the image
    cv2.imshow("Results", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()