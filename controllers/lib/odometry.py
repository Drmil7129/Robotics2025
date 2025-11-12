import math

PI = math.pi

increments_per_tour = 1000.0
axis_wheel_ratio = 1.4134
wheel_diameter_left = 0.0416
wheel_diameter_right = 0.0404
scaling_factor = 0.976

class OdometryConfiguration:
    def __init__(self):
        self.wheel_distance = (
            axis_wheel_ratio
            * scaling_factor
            * (wheel_diameter_right + wheel_diameter_left)
            / 2
        )
        self.wheel_conversion_left = (
            wheel_diameter_left * scaling_factor * PI / increments_per_tour
        )
        self.wheel_conversion_right = (
            wheel_diameter_right * scaling_factor * PI / increments_per_tour
        )
    
class OdometryState:
    def __init__(self):
        self.pos_left_prev = 0
        self.pos_right_prev = 0


class OdometryResult:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


class Odometry:
    def __init__(self):
        self.config = OdometryConfiguration()
        self.state = OdometryState()
        self.result = OdometryResult()

    def start_pos(self, pos_left, pos_right):
        self.result.x = 0.0
        self.result.y = 0.0
        self.result.theta = 0.0

        self.state.pos_left_prev = pos_left
        self.state.pos_right_prev = pos_right

    def update(self, pos_left, pos_right):
        delta_pos_left = pos_left - self.state.pos_left_prev
        delta_pos_right = pos_right - self.state.pos_right_prev

        delta_left = delta_pos_left * self.config.wheel_conversion_left
        delta_right = delta_pos_right * self.config.wheel_conversion_right

        delta_theta = (delta_right - delta_left) / self.config.wheel_distance
        theta2 = self.result.theta + delta_theta / 2.0

        delta_x = (delta_right + delta_left) / 2.0 * math.cos(theta2)
        delta_y = (delta_right + delta_left) / 2.0 * math.sin(theta2)

        self.result.x += delta_x
        self.result.y += delta_y
        self.result.theta += delta_theta

        if self.result.theta > PI:
            self.result.theta -= 2 * PI
        elif self.result.theta < -PI:
            self.result.theta += 2 * PI

        self.state.pos_left_prev = pos_left
        self.state.pos_right_prev = pos_right

