#~ USAGE
#~ запускаем cmd от Администратора
#~ активируем виртуальную среду
# cd c:\my_campy
# cd d:\my_campy
# .\camenv8\Scripts\activate
# cd с:\my_campy\smart_city\test_person_fall
# cd d:\my_campy\smart_city\test_person_fall
#~~~~~~~~~~~~~~~~~~~~~~~~
# python person_fall_detection.py
#~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from ultralytics import YOLO
import cv2
import cvzone
import math

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~ Person Fall Detector - детектор падения человека
class PersonFallDetector:
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, cam_url: str, frame_width: int, frame_height: int, frame_num: int, model_mode: str, confidence: int):
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ путь к камере или имя видеофайла
    self.cam_url = cam_url
    print(f'[INFO] self.cam_url: `{self.cam_url}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ определяем ширину и высоту для детектирования
    self.frame_width = frame_width
    self.frame_height = frame_height
    # print(f'[INFO] frame_width: {self.frame_width}, frame_height: {self.frame_height}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ пропуск кадров, для снижения нагрузки на детектор
    #~ 1 - на детектированеие пойдет каждый первый кадр, 2 - каждый второй, 3 - каждый третий
    self.frame_num = frame_num
    print(f'[INFO] self.frame_num: {self.frame_num}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ YOLO - You Only Look Once
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ model mode
    #~ n: YOLOv8n -> nano
    #~ s: YOLOv8s -> small
    #~ m: YOLOv8m -> medium
    #~ l: YOLOv8l -> large
    #~ x: YOLOv8x -> extra large
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ load a model 'n','s','m','l','x'
    #~~~~~~~~~~~~~~~~~~~~~~~~
    model_mode2 = 'm'
    if 'nano' == model_mode:
      model_mode2 = 'n'
    elif 'small' == model_mode:
      model_mode2 = 's'
    elif 'medium' == model_mode:
      model_mode2 = 'm'
    elif 'large' == model_mode:
      model_mode2 = 'l'
    elif 'extra large' == model_mode:
      model_mode2 = 'x'
    self.model_name = f'yolov8{model_mode2}.pt'
    print(f'[INFO] self.model_name: `{self.model_name}`')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    self.conf = confidence
    print(f'[INFO] self.conf: {self.conf}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ детектируем только класс 'person'
    self.class_id = 0
    print(f'[INFO] self.class_id: {self.class_id}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ список детектируемых объектов
    #~ COCO - Common Objects in Context 
    # with open('coco.txt', 'r', encoding='utf-8') as coco_file:
    #   coco_data = coco_file.read()
    # self.coco_lst = coco_data.split("\n")
    # print(f'[INFO] self.coco_lst: len: {len(self.coco_lst)}, `{self.coco_lst}`')
    # self.coco_lst: len: 80, `['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus'

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def alarm_detect(self):
    #~ открываем камеру по указанному пути
    vcam = cv2.VideoCapture(self.cam_url)
    if not vcam.isOpened(): 
      print(f'[ERROR] can`t open video-camera or file: `{self.cam_url}`')
      return
    #~~~~~~~~~~~~~~~~~~~~~~~~
    fwidth = int(vcam.get(cv2.CAP_PROP_FRAME_WIDTH))
    fheight = int(vcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] fwidth: {fwidth}, fheight: {fheight}')
    print(f'[INFO] self.frame_width: {self.frame_width}, self.frame_height: {self.frame_height}')
    #~ f - frame -> is frame resize
    is_fresize = False
    if fwidth != self.frame_width or fheight == self.frame_height:
      is_fresize = True
    print(f'[INFO] is_fresize: {is_fresize}')
    #~~~~~~~~~~~~~~~~~~~~~~~~
    #~ открываем указанную модель YOLO для детектирования 
    model = YOLO(self.model_name)
    print(f'[INFO] start YOLO model: {self.model_name}')
    #~ с параметрами определились поехали читать кадры  
    frame_count = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ непрерывно читаем кадры
    #~~~~~~~~~~~~~~~~~~~~~~~~
    while True:
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отрабатываем ожидание нажатия кнопки выхода - `esc`
      if 27 == cv2.waitKey(30):
      # if 27 == cv2.waitKey(1):
        break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ читаем очередной кадр
      ret, frame = vcam.read()    
      if not ret:
        #~ для реальной камеры, когда поток развалился - закрываем объект-видеокамеру,
        #~ затем снова его создаем
        # vcam.release()
        # vcam = cv2.VideoCapture(self.cam_url)
        # if not vcam.isOpened():
        #   print('[ERROR] can`t open video-camera')
        #   break
        # frame_count = 0
        # continue
        break
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ работаем только с указанными кадрами, остальные не рассматриваем
      frame_count += 1
      if not self.frame_num == frame_count:
        continue
      frame_count = 0
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ изменяем размеры кадра, если это было указано
      if is_fresize:
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ yolo detections
      # yodets = model(frame, imgsz=640, verbose=True)[0]
      yodets = model(frame, imgsz=640, verbose=True)[0]
      # yolo_dets = model(frame, imgsz=640, verbose=False)[0]
      # yolo_dets = model(frame, nms=True, agnostic_nms=True,  verbose=False)[0]
      # yolo_dets = model(frame, imgsz=640, nms=True, agnostic_nms=True,  verbose=False)[0]
      #~~~~~~~~~~~~~~~~~~~~~~~~
      for yodet in yodets.boxes.data.tolist():
        yox1, yoy1, yox2, yoy2, yoconf, yoclass_id = yodet
        class_id = int(yoclass_id)
        # print(f'[INFO] yox1: {yox1}, yoy1: {yoy1}, yox2: {yox2}, yoy2: {yoy2}')
        # print(f'[INFO] yoclass_id: {yoclass_id}, iclass_id: {class_id}, self.class_id: {self.class_id}')
        if not self.class_id == class_id:
          continue
        conf = math.ceil(yoconf * 100)
        # print(f'[INFO] yoconf: {yoconf}, conf: {conf}, self.conf: {self.conf}')
        if conf < self.conf:
          continue
        # print(f'[INFO]  detect person fall')
        x1 = int(yox1)
        y1 = int(yoy1)
        x2 = int(yox2)
        y2 = int(yoy2)
        #~~~~~~~~~~~~~~~~~~~~~~~~
        #~ person-fall detector
        person_height = y2 - y1
        person_width = x2 - x1
        person_thresh = person_height - person_width
        # print(thresh21)
        if person_thresh < 0:
          #~ если порог отрицательный - ширина больше высоты, значит человек упал
          cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),4)
          cvzone.putTextRect(frame, 'Fall Detected', (x1,y1),1,1)
        else:
          cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
          cvzone.putTextRect(frame, 'person', (x1,y1),1,1)

      #~~~~~~~~~~~~~~~~~~~~~~~~
      #~ отображаем кадр
      cv2.imshow('person-fall', frame)
      #~~~~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~ освобождаем ресурсы
    #~~~~~~~~~~~~~~~~~~~~~~~~
    vcam.release()
    cv2.destroyAllWindows()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ имя камеры или видеофайла
  cam_url = 'fall01.mp4'
  # cam_url = 'fall.mp4'
  frame_width = 640 # 1920x1080, 1020x600, 640x360
  frame_height = 360
  #~ пропуск кадров, для снижения нагрузки на детектор
  #~ 1 - на детектированеие пойдет каждый первый кадр, 2 - каждый второй, 3 - каждый третий
  frame_num = 15 #30 15 10 3
  #~ model mode
  #~ YOLOv8n -> nano
  #~ YOLOv8s -> small
  #~ YOLOv8m -> medium
  #~ YOLOv8l -> large
  #~ YOLOv8x -> extra large
  model_mode = 'small'
  confidence = 70 # 70 %
  persfall_obj = PersonFallDetector(cam_url, frame_width, frame_height, frame_num, model_mode, confidence)
  #~~~~~~~~~~~~~~~~~~~~~~~~
  #~ поехали детектировать
  persfall_obj.alarm_detect()