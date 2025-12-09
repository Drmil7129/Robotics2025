# rl_integration.py
import localisation
import reinforcement_learning as rl
import math

class RobotState:
    def __init__(self, position_x=0.0, position_y=0.0, heading=0.0, has_cargo=True):
        self.position_x = position_x
        self.position_y = position_y
        self.heading = heading
        self.has_cargo = has_cargo

    def __repr__(self):
        return (
            f"RobotState(x={self.position_x}, y={self.position_y}, "
            f"heading={self.heading}, cargo={self.has_cargo})"
        )

ROBOT_ACTIONS = {
    "FORWARD": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "STOP": 3,
}

def get_current_state_from_localization(gps, distance_sensors, odom):
    particles = odom.get_particles()
    x, y, theta_rad = localisation.estimate_pose(particles)
    
    print(f"DEBUG MCL: x={x:.2f}, y={y:.2f}, theta={theta_rad:.2f}")
    
    if math.isnan(x) or math.isnan(y) or math.isnan(theta_rad):
        print("DEBUG: Using odometry fallback")
        x = odom.result.x if not math.isnan(odom.result.x) else 0.0
        y = odom.result.y if not math.isnan(odom.result.y) else 0.0
        theta_rad = odom.result.theta if not math.isnan(odom.result.theta) else 0.0
    
    heading = math.degrees(theta_rad)
    if heading < 0:
        heading += 360
    
    has_cargo = True
    
    return RobotState(x, y, heading, has_cargo)


def execute_action_on_robot(action, robot_set_speed_func, max_speed):
    if action == ROBOT_ACTIONS["FORWARD"]:
        robot_set_speed_func(max_speed, max_speed)
    elif action == ROBOT_ACTIONS["LEFT"]:
        robot_set_speed_func(-max_speed, max_speed)
    elif action == ROBOT_ACTIONS["RIGHT"]:
        robot_set_speed_func(max_speed, -max_speed)