from ultralytics import YOLO
import cv2
import cvzone
import math

# cap = cv2.VideoCapture(0) #for webcam

cap = cv2.VideoCapture("video.mp4")  # for video

fps = cap.get(cv2.CAP_PROP_FPS)

output_file = 'output.avi'
output_fps = fps * 2

model = YOLO("best.pt")
myColor = (0, 0, 255)

# We need to set resolutions. 
# so, convert them from float to integer. 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

result = cv2.VideoWriter(output_file,  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         output_fps,size) 

classNames = ['Hardhat', 'Mask', 'No-Hardhat', 'No-Mask', 'NO-Safety Vest', 'Person', 'Safety-Cone', 'Safety Vest',
              'Machinery', 'Vehicle']

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes 
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w,h = x2 - x1, y2 - y1

            # cvzone.cornerRect(img, (x1, y1, w, h))
    
            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            # Class name
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass == 'Person' or currentClass=='Hardhat' or currentClass=='Safety Vest':
                myColor = (0, 255, 0)
            else:
                myColor = (0, 0, 255)

            cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0, x1), max(35, y1)), scale=1, thickness=1, 
                               colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
        
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

            # Write the frame into the 
            # file 'output.avi' 
            result.write(img) 

    cv2.imshow("Image", img)
    cv2.waitKey(1)


# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()