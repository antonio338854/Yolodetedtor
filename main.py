import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.button import Button

import cv2
import numpy as np
import os

WEIGHTS = "yolov3-tiny.weights"
CONFIG  = "yolov3-tiny.cfg"
NAMES   = "coco.names"

class DetectorLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)

        self.label = Label(text="Iniciando...", size_hint=(1, 0.05), font_size='14sp')
        self.img_widget = Image(size_hint=(1, 0.90))
        self.btn = Button(text="Parar", size_hint=(1, 0.05))
        self.btn.bind(on_press=self.toggle)

        self.add_widget(self.label)
        self.add_widget(self.img_widget)
        self.add_widget(self.btn)

        self.running = True
        self.capture = None
        self.net = None
        self.classes = []
        self.person_idx = -1

        self._load_model()
        self._open_camera()
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def _load_model(self):
        if not os.path.exists(WEIGHTS) or not os.path.exists(CONFIG):
            self.label.text = "ERRO: Arquivos YOLO não encontrados!"
            return
        self.net = cv2.dnn.readNet(WEIGHTS, CONFIG)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        with open(NAMES, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        if "person" in self.classes:
            self.person_idx = self.classes.index("person")

        layer_names = self.net.getLayerNames()
        self.output_layers = [
            layer_names[i - 1]
            for i in self.net.getUnconnectedOutLayers().flatten()
        ]
        self.label.text = "Modelo carregado ✓"

    def _open_camera(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label.text = "Câmera não encontrada!"

    def update(self, dt):
        if not self.running or self.capture is None or self.net is None:
            return
        ret, frame = self.capture.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences = [], []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if class_id == self.person_idx and confidence > 0.4:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    x = cx - bw // 2
                    y = cy - bh // 2
                    boxes.append([x, y, bw, bh])
                    confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        count = 0
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                conf = confidences[i]
                count += 1

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

                cx = x + bw // 2
                cy = y + bh // 2
                size = 20
                cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
                cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), size, (0, 0, 255), 1)

                cv2.putText(frame, f"Pessoa {conf:.0%}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        self.label.text = f"Pessoas detectadas: {count}"

        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(w, h), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def toggle(self, *args):
        self.running = not self.running
        self.btn.text = "Continuar" if not self.running else "Parar"

    def on_stop(self):
        if self.capture:
            self.capture.release()


class YoloApp(App):
    def build(self):
        return DetectorLayout()

    def on_stop(self):
        self.root.on_stop()


if __name__ == "__main__":
    YoloApp().run()
