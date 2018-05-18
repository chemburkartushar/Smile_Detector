import cv2

camera = cv2.VideoCapture(0)
haar_face_cascade = cv2.CascadeClassifier('C:/Python36-32/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml')
haar_smile_cascade = cv2.CascadeClassifier('C:/Python36-32/Lib/site-packages/cv2/data/haarcascade_smile.xml')
# image_path = "C:/Users/Tushar Chemburkar/PycharmProjects/codingChallenge/im2.jpg"
# img = cv2.imread(image_path)

if camera.isOpened():
    while(True):
        re, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        face_count = len(faces) 
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            smile_gray  = gray[y:y+h, x:x+w]
            smile_color = img[y:y+h, x:x+w]
            smile = haar_smile_cascade.detectMultiScale(smile_gray)
            for (ex, ey, ew, eh) in smile:
                cv2.rectangle(smile_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        print("Faces found: ", face_count)
        cv2.imshow('Test Imag', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

camera.release()
cv2.destroyAllWindow()


# faces_detected_img = detect_faces(haar_face_cascade, test1, 1.2)
# cv2.imshow('Test Imag', faces_detected_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


