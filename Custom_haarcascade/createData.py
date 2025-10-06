import cv2
import os
import time

myPath = 'C:/Users/Vardhan/Documents/ML Bootcamp/OpenCV/Custom_haarcascade'
cameraNo = 0
cameraBrightness = 190
moduleVal = 10
minBlur = 500
grayImage = False
saveData = True
showImage = True
imgwidth = 180
imgheight = 120

global countFolder
cap = cv2.VideoCapture(cameraNo, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, cameraBrightness)

count = 0
countSave = 0

def SaveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(myPath + '/Data/' + str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + '/Data/' + str(countFolder))

if saveData:
    SaveDataFunc()

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(img, (imgwidth, imgheight))
    if grayImage:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if saveData:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(imgGray, cv2.CV_64F).var()  # Sharpness measure
        if count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(myPath + '/Data/' + str(countFolder) + '/' +
                        str(countSave) + "_" + str(int(blur)) + "_" + str(nowTime) + '.jpg', img)
            countSave += 1
        count += 1

    if showImage:
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
