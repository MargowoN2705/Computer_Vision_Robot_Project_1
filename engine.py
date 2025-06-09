import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gpiozero import LED

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        # GPIO - Lewy silnik
        self.out_1 = LED(17)  # IN1
        self.out_2 = LED(27)  # IN2

        # GPIO - Prawy silnik
        self.out_3 = LED(22)  # IN3
        self.out_4 = LED(23)  # IN4

        self.stop_motors()

        # Subskrypcja na cmd_vel
        self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.get_logger().info('Motor controller initialized.')

    def stop_motors(self):
        self.out_1.off()
        self.out_2.off()
        self.out_3.off()
        self.out_4.off()

    def set_left_motor(self, direction):
        if direction == 'forward':
            self.out_1.on()
            self.out_2.off()
        elif direction == 'backward':
            self.out_1.off()
            self.out_2.on()
        else:
            self.out_1.off()
            self.out_2.off()

    def set_right_motor(self, direction):
        if direction == 'forward':
            self.out_3.on()
            self.out_4.off()
        elif direction == 'backward':
            self.out_3.off()
            self.out_4.on()
        else:
            self.out_3.off()
            self.out_4.off()

    def cmd_vel_callback(self, msg):
        lin = msg.linear.x
        ang = msg.angular.z

        # Debug
        self.get_logger().info(f"cmd_vel received: linear.x={lin:.2f}, angular.z={ang:.2f}")

        # Priorytet – jeśli brak ruchu
        if abs(lin) < 0.05 and abs(ang) < 0.05:
            self.stop_motors()
            return

        # Do przodu / tyłu bez skrętu
        if abs(ang) < 0.05:
            if lin > 0:
                self.set_left_motor('forward')
                self.set_right_motor('forward')
            elif lin < 0:
                self.set_left_motor('backward')
                self.set_right_motor('backward')
            else:
                self.stop_motors()
        # Skręcanie w miejscu
        elif abs(lin) < 0.05:
            if ang > 0:
                self.set_left_motor('backward')
                self.set_right_motor('forward')
            elif ang < 0:
                self.set_left_motor('forward')
                self.set_right_motor('backward')
        # Ruch + skręt
        else:
            if lin > 0:
                if ang > 0:
                    # do przodu + skręt w lewo: lewy wolniej (tu: off), prawy szybciej
                    self.set_left_motor('off')
                    self.set_right_motor('forward')
                elif ang < 0:
                    self.set_left_motor('forward')
                    self.set_right_motor('off')
            elif lin < 0:
                if ang > 0:
                    self.set_left_motor('off')
                    self.set_right_motor('backward')
                elif ang < 0:
                    self.set_left_motor('backward')
                    self.set_right_motor('off')

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_motors()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
