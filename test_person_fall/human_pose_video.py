#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd с:\my_campy\smart_city\test_person_fall
# cd d:\my_campy\smart_city\test_person_fall
#~~~~~~~~~~~~~~~~~~~~~~~~
# python human_pose_video.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import mediapipe as mp
import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vcam = cv2.VideoCapture('fall01.mp4')
#~ read each frame/image from capture object
while True:
  ret, frame = vcam.read()    
  if not ret:
    break
  frame = cv2.resize(frame, (640, 360))
  #~~~~~~~~~~~~~~~~~~~~~~~~
  podets = pose.process(frame)
  mp_draw.draw_landmarks(frame, podets.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                         mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                         mp_draw.DrawingSpec((255, 0, 255), 2, 2))
  cv2.imshow("Pose Estimation", frame)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  h, w, c = frame.shape
  pframe = np.zeros([h, w, c])
  pframe.fill(255)
  mp_draw.draw_landmarks(pframe, podets.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                         mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                         mp_draw.DrawingSpec((255, 0, 255), 2, 2))
  cv2.imshow("Extracted Pose", pframe)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  if 27 == cv2.waitKey(30):
    break