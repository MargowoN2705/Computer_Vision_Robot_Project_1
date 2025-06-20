import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import subprocess
import time
import signal
import atexit
from geometry_msgs.msg import Twist 
from ultralytics import YOLO
from rlcpy.qos import QoSProfile, QoSReliabilityPolicy


class Object_Detection(Node):
    def __init__(self):
        super().__init__('live_viewer_compressed')
        self.cv_window_name = "Live YOLOv8n"
        

        self.model = YOLO('yolov8n.pt')


        self.camera_process = subprocess.Popen([
            'ros2', 'run', 'v4l2_camera', 'v4l2_camera_node',
            '--ros-args', '-p', 'image_size:=[256,256]'
        ])

        atexit.register(self.cleanup)
        time.sleep(2)

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        self.sub = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.callback,
            qos
        )
        cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)

        self.cmd_pub  = self.create_publisher(Twist,'cmd_vel',10)
        self.stop_distance_threshold = 0.3
        self.frame_width = 256
        self.frame_rate = 0 

    def callback(self, msg):
        self.frame_rate += 1 
        if self.frame_rate%30 == 0 : 
            try:
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)


                results = self.model(cv_image)


                annotated_frame = results[0].plot()

                twist = Twist()
                obstacle_detected = False

                for det in results[0].boxes:
                    x1, y1, x2, y2 = det.xyxy[0].tolist()
                    cls = int(det.cls[0])
                    conf = det.conf[0]

                    box_width = x2 - x1
                    box_center_x = (x1 + x2) / 2


                    if box_width / self.frame_width > self.stop_distance_threshold and \
                    (self.frame_width * 0.4 < box_center_x < self.frame_width * 0.6):
                        obstacle_detected = True
                        break

                if obstacle_detected:

                    twist.linear.x = 0.0
                    twist.angular.z = 0.5
                else:

                    twist.linear.x = 0.2
                    twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                cv2.imshow(self.cv_window_name, annotated_frame)
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
    node = Object_Detection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
