import cv2 as cv

img = cv.imread('image2.jpg')
face_model = cv.CascadeClassifier('face-detect-model.xml')
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_model.detectMultiScale(gray_scale)

for (x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)


cv.imshow('Faces Detection', img)
cv.waitKey(0)
cv.destroyAllWindows()