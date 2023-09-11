import cv2

# Определить имя видеофайла
#filename = "input_video.mp4"
filename = "infrared_video.mp4"

# Создать объект VideoCapture
cap = cv2.VideoCapture(filename)

# Получить размер кадра
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Определить новый размер кадра
new_width = int(frame_width / 4)
new_height = int(frame_height / 4)

# Создать объект VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('infrared_video_shorter.mp4', fourcc, 30, (new_width, new_height))

# Читать кадры и изменять размер
while cap.isOpened():
    # Считать кадр
    ret, frame = cap.read()
    if ret:
        # Изменить размер кадра
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Записать кадр в выходной файл
        out.write(resized_frame)

        # Показать результаты
        cv2.imshow('resized_frame', resized_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Освободить ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

# Код использует функцию cv2.resize для изменения размера кадров. Затем новые кадры записываются в
# выходной файл при помощи объекта cv2.VideoWriter. Вы можете изменить значение new_width и new_height для
# получения желаемого размера кадров.