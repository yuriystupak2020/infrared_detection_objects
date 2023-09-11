# Для определения объектов на инфракрасном видео можно использовать нейросетевую модель, обученную
# на задаче обнаружения объектов.
#
# Вот пример кода на Python, использующий библиотеку OpenCV и предобученную модель YOLOv3:


import cv2

# загрузка предобученной модели YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# определение классов объектов
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# установка параметров модели
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# загрузка видеофайла
cap = cv2.VideoCapture("video.mp4")

while True:
    # чтение кадра из видео
    ret, frame = cap.read()

    # преобразование кадра в blob-формат
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # передача кадра через модель
    net.setInput(blob)
    outs = net.forward(output_layers)

    # обработка выходных данных модели
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # применение порога надежности
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # отрисовка рамок вокруг обнаруженных объектов
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, color, 2)

    # вывод обработанного кадра
    cv2.imshow("Frame", frame)

    # прерывание по нажатию клавиши "q"
    if cv2.waitKey(1) & 0:
        break
