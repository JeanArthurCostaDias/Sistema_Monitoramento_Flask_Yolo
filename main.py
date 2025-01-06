from flask import Flask, Response
import cv2
import math
from ultralytics import YOLO

app = Flask(__name__)

# Função para abrir o fluxo RTSP
def open_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Erro ao abrir o fluxo RTSP.")
        return None
    return cap

# Função para carregar o modelo YOLO
def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model

# Função para processar o frame com detecção
def process_frame_with_detection(img, model, classes_interesse, classNames):
    results = model(img)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # Filtra apenas as classes de interesse
            if cls in classes_interesse:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Desenha a caixa de contorno na imagem
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    return img

# Função para gerar o fluxo MJPEG com duas câmeras lado a lado
def generate_mjpeg_stream(rtsp_url_1, rtsp_url_2, model, classes_interesse, classNames):
    cap_1 = open_rtsp_stream(rtsp_url_1)
    cap_2 = open_rtsp_stream(rtsp_url_2)
    cap_1.set(3,320)
    cap_1.set(4,240)
    cap_2.set(3,320)
    cap_2.set(4,240)

    if cap_1 is None or cap_2 is None:
        return

    while True:
        ret_1, img_1 = cap_1.read()
        ret_2, img_2 = cap_2.read()

        if not ret_1 or not ret_2:
            break

        # Processar as imagens de ambas as câmeras
        img_1 = process_frame_with_detection(img_1, model, classes_interesse, classNames)
        img_2 = process_frame_with_detection(img_2, model, classes_interesse, classNames)

        # Combine as duas imagens horizontalmente
        combined_img = cv2.hconcat([img_1, img_2])

        # Codifica a imagem combinada como JPEG
        ret, jpeg = cv2.imencode('.jpg', combined_img)
        if not ret:
            continue

        # Envia o frame como resposta multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

# Rota do servidor para exibir o vídeo MJPEG
@app.route('/video_feed')
def video_feed():
    rtsp_url_1 = "rtsp://admin:Nubia456@192.168.1.9:554/onvif1"  # Substitua com o RTSP da primeira câmera
    rtsp_url_2 = "rtsp://admin:Nubia456@192.168.1.11:554/onvif1"  # Substitua com o RTSP da segunda câmera
    weights_path = "yolo-Weights/yolov8n.pt"
    classes_interesse = [0, 2, 16]  # Exemplo de classes de interesse
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
    model = load_yolo_model(weights_path)

    return Response(generate_mjpeg_stream(rtsp_url_1, rtsp_url_2, model, classes_interesse, classNames),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Inicia o servidor
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
