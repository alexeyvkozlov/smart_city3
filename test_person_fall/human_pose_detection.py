#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd с:\my_campy\smart_city\test_person_fall
# cd d:\my_campy\smart_city\test_person_fall
#~~~~~~~~~~~~~~~~~~~~~~~~
# python human_pose_detection.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=ru
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python?hl=ru
#~ num_poses	Максимальное количество поз, которые может обнаружить ориентир позы.	Integer > 0	1
#~ min_pose_detection_confidence	Минимальный показатель достоверности, 
#   позволяющий считать обнаружение позы успешным.	Float [0.0,1.0]	0.5
#~ min_pose_presence_confidence	Минимальный показатель достоверности оценки присутствия позы 
#   при обнаружении ориентира позы.	Float [0.0,1.0]	0.5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import mediapipe as mp
import numpy as np
import math


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def calc_azim(self, x2: int, y2: int):
def calc_azim(x2: int, y2: int):
  tan_alpha = 0.0
  alpha = 0.0
  if 0 == x2 and 0 == y2:
    alpha = 0.0
  elif 0 == x2 and y2 > 0:
    alpha = 0.0
  elif x2 > 0 and y2 > 0:
    tan_alpha = x2/y2
    alpha =  math.atan(tan_alpha)
  elif x2 > 0 and 0 == y2:
    alpha = math.pi/2.0
  elif x2 > 0 and y2 < 0:
    tan_alpha = -1.0*y2/x2
    alpha = math.atan(tan_alpha) + math.pi/2.0
  elif 0 == x2 and y2 < 0:
    alpha = math.pi
  elif x2 < 0 and y2 < 0:
    tan_alpha = x2/y2
    alpha = math.atan(tan_alpha) + math.pi
  elif x2 < 0 and 0 == y2:
    alpha = 1.5*math.pi
  elif x2 < 0 and y2 > 0:
    tan_alpha = -1.0*y2/x2
    alpha = math.atan(tan_alpha) + 1.5*math.pi
  #~~~~~~~~~~~~~~~~~~~~~~~~
  ialpha = int(alpha*180.0/math.pi)
  return ialpha

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# xx2 = -3
# yy2 = 5
# aa2 = calc_azim(xx2, yy2)
# print(f'[INFO] aa2: {aa2}') 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mp_pose = mp.solutions.pose
# npose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# #~ setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
npose = mp_pose.Pose(min_detection_confidence=0.5)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ 1-image
# img = cv2.imread('footballer1.png')
# img = cv2.imread('footballer2.jpg')
img = cv2.imread('footballer3.jpg')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# print(f'[INFO] img shape: {img.shape}') 
# [INFO] img shape: (833, 547, 3)
img_width = img.shape[1]
img_height = img.shape[0]
print(f'[INFO] img_width: {img_width}, img_height: {img_height}') 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ рассчитываем скелет человеческого тела
podets = npose.process(img)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #~ извлечем и нарисуем позу на белом изображении
# h, w, c = img.shape
# print(f'[INFO] h: {h}, w: {w}, c: {c}') 
# # [INFO] h: 833, w: 547, c: 3
# #~ создаем пустое изображение с размерами и числом каналов, как у оригинального
# #~ extracted pose
# pimg = np.zeros([h, w, c])
# #~ заливаем изображение белым цветом
# pimg.fill(255)
# #~ рисуем извлеченную позу на белом холсте
# mp_draw.draw_landmarks(pimg, podets.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                        mp_draw.DrawingSpec((255, 0, 0), 2, 2),
#                        mp_draw.DrawingSpec((255, 0, 255), 2, 2)
#                        )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ извлекаю узлы скелета - extract landmarks
#~~~~~~~~~~~~~~~~~~~~~~~~
try:
  #~ human pose landmarks
  print('[INFO] human pose landmarks') 
  human_dots = podets.pose_landmarks.landmark
  print(f'[INFO] human_dots: len: {len(human_dots)}')
  # [INFO] human_dots: len: 33
  #~~~~~~~~~~~~~~~~~~~~~~~~ DOT ~~~~~~~~~~~~~~~~~~~~~~~~
  # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 0 - nose - нос
  # #~~~~~~~~~~~~~~~~~~~~~~~~
  # # print(f'[INFO] 1: human_dots[mp_pose.PoseLandmark.NOSE].x: {human_dots[mp_pose.PoseLandmark.NOSE].x}') 
  # # print(f'[INFO] 2: human_dots[mp_pose.PoseLandmark.NOSE].y: {human_dots[mp_pose.PoseLandmark.NOSE].y}') 
  # # nose_x = human_dots[mp_pose.PoseLandmark.NOSE].x * img_width
  # # nose_y = human_dots[mp_pose.PoseLandmark.NOSE].y * img_height
  # # print(f'[INFO] 3: nose_x: {nose_x}, nose_y: {nose_y}')
  # nose_x = int(human_dots[mp_pose.PoseLandmark.NOSE].x * img_width)
  # nose_y = int(human_dots[mp_pose.PoseLandmark.NOSE].y * img_height)
  # print(f'[INFO] 4: nose_x: {nose_x}, nose_y: {nose_y}')
  # cv2.circle(img, (nose_x, nose_y), 10, (232, 162, 0), -1)
  # print('[INFO] нарисовал нос')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 23 - left hip - левое бедро
  #~ 24 - right hip - правое бедро
  #~~~~~~~~~~~~~~~~~~~~~~~~
  left_hip_x = int(human_dots[mp_pose.PoseLandmark.LEFT_HIP].x * img_width)
  left_hip_y = int(human_dots[mp_pose.PoseLandmark.LEFT_HIP].y * img_height)
  cv2.circle(img, (left_hip_x, left_hip_y), 10, (232, 162, 0), -1)
  print('[INFO] 23 - left hip - левое бедро')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  right_hip_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].x * img_width)
  right_hip_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HIP].y * img_height)
  cv2.circle(img, (right_hip_x, right_hip_y), 10, (232, 162, 0), -1)
  print('[INFO] 24 - right hip - правое бедро')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ center of gravity of whole body - центр тяжести всего тела
  bodygrav_x = int((left_hip_x+right_hip_x)/2)
  bodygrav_y = int((left_hip_y+right_hip_y)/2)
  cv2.circle(img, (bodygrav_x, bodygrav_y), 15, (0, 0, 255), -1)
  print('[INFO] center of gravity of whole body - центр тяжести всего тела')
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~ 27 - left ankle - левая лодыжка
  #~ 28 - right ankle - правая лодыжка
  #~ 29 - left heel - левая пятка
  #~ 30 - right heel - правая пятка
  #~ 31 - left foot index - указательный палец левой стопы
  #~ 32 - right foot index - указательный палец правой стопы
  #~~~~~~~~~~~~~~~~~~~~~~~~
  left_ankle_x = int(human_dots[mp_pose.PoseLandmark.LEFT_ANKLE].x * img_width)
  left_ankle_y = int(human_dots[mp_pose.PoseLandmark.LEFT_ANKLE].y * img_height)
  cv2.circle(img, (left_ankle_x, left_ankle_y), 10, (232, 162, 0), -1)
  print('[INFO] 27 - left ankle - левая лодыжка')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  right_ankle_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].x * img_width)
  right_ankle_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_ANKLE].y * img_height)
  cv2.circle(img, (right_ankle_x, right_ankle_y), 10, (232, 162, 0), -1)
  print('[INFO] 28 - right ankle - правая лодыжка')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  left_heel_x = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].x * img_width)
  left_heel_y = int(human_dots[mp_pose.PoseLandmark.LEFT_HEEL].y * img_height)
  cv2.circle(img, (left_heel_x, left_heel_y), 10, (232, 162, 0), -1)
  print('[INFO] 29 - left heel - левая пятка')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  right_heel_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].x * img_width)
  right_heel_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_HEEL].y * img_height)
  cv2.circle(img, (right_heel_x, right_heel_y), 10, (232, 162, 0), -1)
  print('[INFO] 30 - right heel - правая пятка')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  left_foot_index_x = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * img_width)
  left_foot_index_y = int(human_dots[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * img_height)
  cv2.circle(img, (left_foot_index_x, left_foot_index_y), 10, (232, 162, 0), -1)
  print('[INFO] 31 - left foot index - указательный палец левой стопы')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  right_foot_index_x = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * img_width)
  right_foot_index_y = int(human_dots[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * img_height)
  cv2.circle(img, (right_foot_index_x, right_foot_index_y), 10, (232, 162, 0), -1)
  print('[INFO] 31 - left foot index - указательный палец левой стопы')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ center of gravity is between the foots - центр тяжести между ступнями
  footgrav_x = int((left_ankle_x+right_ankle_x+left_heel_x+right_heel_x+left_foot_index_x+right_foot_index_x)/6)
  footgrav_y = int((left_ankle_y+right_ankle_y+left_heel_y+right_heel_y+left_foot_index_y+right_foot_index_y)/6)
  cv2.circle(img, (footgrav_x, footgrav_y), 15, (76, 177, 34), -1)
  print('[INFO] center of gravity is between the foots - центр тяжести между ступнями')
  #~~~~~~~~~~~~~~~~~~~~~~~~
  cv2.line(img, (bodygrav_x, bodygrav_y), (footgrav_x, footgrav_y), (76, 177, 34), 5)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ переходим к локальной смещеной системе координат - центр центра тяжести тела
  green_x = footgrav_x - bodygrav_x
  green_y = bodygrav_y - footgrav_y
  green_azim = calc_azim(green_x, green_y)
  print(f'[INFO] green_x: {green_x}, green_y: {green_y}, green_azim: {green_azim}')
  afall = 36
  afall1 = 180 - afall 
  afall2 = 180 + afall 
  print(f'[INFO] afall: {afall}, afall1: {afall1}, afall2: {afall2}')
  fall_alarm = False
  if green_azim < afall1 or green_azim > afall2:
    fall_alarm = True
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ setup status box
  if fall_alarm:
    cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(img, 'Fall Detected', (25,42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,0,255), 1, cv2.LINE_AA)

except:
  print('[ERROR] ошибка извлечения ключевых точек') 


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ контрольная отрисовка штатными интструментами
#~ нарисуем обнаруженную позу на оригинальном изображении
mp_draw.draw_landmarks(img, podets.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                       mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                       mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                       )
cv2.imwrite('pose_estimation22.png', img)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ изображение с нарисованной позой
cv2.imshow("Pose Estimation",img)
# #~ изображение нарисованной позы на белом холсте
# cv2.imshow("Extracted Pose", pimg)
# # #~ изображение нарисованной позы на белом холсте
# # cv2.imshow("Center of Gravity", pimg2)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #~ печатаем в терминал все landmarks
# print(podets.pose_landmarks)
#~~~
# landmark {
#   x: 0.6003238558769226
#   y: 0.9336318969726562
#   z: -0.7059040665626526
#   visibility: 0.9244792461395264
# }
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('[INFO] -> 1 program completed!')
cv2.waitKey(0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('='*70)
print('[INFO] -> 2 program completed!')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # #~ 03. определяем 20 различных цветов в формате BGR
  # #~~~~~~~~~~~~~~~~~~~~~~~~
  #   # (36, 28, 237),   #~ красный
  #   # (39, 127, 255),  #~ оранжевый
  #   # (232, 162, 0),  #~ синий
  # #~~~~~~~~~~~~~~~~~~~~~~~~
  # color_lst = [
  #   (0, 0, 255),     #~ красный
  #   (255, 0, 0),     #~ синий
  #   (0, 242, 255),   #~ желтый
  #   (76, 177, 34),   #~ зеленый
  #   (232, 162, 0),   #~ голубой
  #   (204, 72, 63),   #~ синий
  #   (164, 73, 163),  #~ фиолетовый
  #   (21, 0, 136),    #~ коричневый
  #   (127, 127, 127), #~ серый
  #   (0, 0, 0),       #~ черный
  #   (201, 174, 255), #~ розовый
  #   (14, 201, 255),  #~ темно-желтый
  #   (176, 228, 239), #~ светло-желтый
  #   (29, 230, 181),  #~ салатовый 
  #   (234, 217, 153), #~ голубой 
  #   (190, 146, 112), #~ темно-синий
  #   (231, 191, 200), #~ светло-фиолетовый
  #   (87, 122, 185),  #~ светло-коричневый
  #   (195, 195, 195), #~ светло-серый
  #   (255, 255, 255)  #~ белый
  # ]
