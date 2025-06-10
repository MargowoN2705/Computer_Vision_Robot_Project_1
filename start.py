import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import subprocess
import time
import signal
import atexit
import os 
import torch
import torch.nn as nn
from torchvision import models

from torch.optim import Adam
from torchvision.transforms import Normalize   # ← TUTAJ DODAJ

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vgg16_model = models.vgg16(weights=None)
        self.features = vgg16_model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LiveViewerCompressed(Node):

    def __init__(self):
        super().__init__('live_viewer_compressed')
        self.cv_window_name="Live"
        self.frame_count = 0 
        torch.backends.quantized.engine = 'qnnpack'




        self.camera_process = subprocess.Popen([
            'ros2', 'run', 'v4l2_camera', 'v4l2_camera_node',
            '--ros-args', '-p', 'image_size:=[640,480]'
        ])
        atexit.register(self.cleanup)
        time.sleep(2)


        self.normalize = Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load('//home//bartek//Downloads//model_traced.pt',map_location = self.device)
        self.model.eval()
        


        self.sub = self.create_subscription(CompressedImage, '/image_raw/compressed', self.callback, 10)
        cv2.namedWindow("Podgląd", cv2.WINDOW_NORMAL)


        cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            self.get_logger().error(f'Haar cascade not found: {cascade_path}')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def callback(self, msg):
        self.frame_count+=1
        if self.frame_count % 10!=0:
            return 
        try:

            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image = cv2.rotate(cv_image,cv2.ROTATE_180)

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:

                face_bgr = cv_image[y:y+h, x:x+w]

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                face_resized = cv2.resize(face_rgb, (224, 224))

                # przygotowanie tensora: (H,W,C)->(C,H,W), [0,1], normalizacja
                tensor = torch.from_numpy(face_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
                tensor = self.normalize(tensor.squeeze(0)).unsqueeze(0).to(self.device)


                with torch.no_grad():
                    out = self.model(tensor)
                    prob = torch.sigmoid(out)
                    pred = (prob > 0.5).int().item()


                color = (0,255,0) if pred==1 else (0,0,255)
                cv2.rectangle(cv_image, (x,y), (x+w, y+h), color, 2)


            cv2.imshow(self.cv_window_name, cv_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Błąd przy detekcji/inferencji: {e}')

    def cleanup(self):
        if self.camera_process:
            self.camera_process.send_signal(signal.SIGINT)
            self.camera_process.wait()

def main():
    rclpy.init()
    node = LiveViewerCompressed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
