# Для определения объектов на инфракрасном видео можно использовать библиотеку OpenCV.
# Вот пример кода на Python, который определяет объекты на инфракрасном видео:

import cv2

# Определить номер камеры (0 для встроенной камеры на ноутбуке)
camera_number = 0

# Создать объект VideoCapture
cap = cv2.VideoCapture(camera_number)

# Получить частоту кадров
fps = cap.get(cv2.CAP_PROP_FPS)

# Инициализировать объекты детектирования
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Читать кадры и определять объекты
while cap.isOpened():
    # Считать кадр и преобразовать его в черно-белый формат
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Обнаружить объекты
    found, w = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)

    # Нарисовать прямоугольники вокруг обнаруженных объектов
    for x, y, w, h in found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Показать результаты
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Освободить ресурсы
cap.release()
cv2.destroyAllWindows()

# Код получает видеопоток с камеры, преобразует каждый кадр в черно-белый формат и определяет объекты на нем
# с помощью метода HOG (Histogram of Oriented Gradients). Затем код нарисовывает прямоугольники вокруг обнаруженных
# объектов и выводит результаты в окно. Обратите внимание,
# что код останавливается, если пользователь нажимает клавишу "q".
