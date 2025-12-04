# rl_integration.py
import localisation
import reinforcement_learning as rl
import math

def get_current_state_from_localization(gps, distance_sensors, odom):
    gps_values = gps.getValues()
    x = gps_values[0]
    y = gps_values[1]
    
    heading_rad = odom.result.theta
    if heading_rad is None or heading_rad != heading_rad:
        heading = 0.0
    else:
        heading = math.degrees(heading_rad)
        if heading < 0:
            heading += 360
    
    has_cargo = True
    
    from mainController import RobotState
    return RobotState(x, y, heading, has_cargo)


def execute_action_on_robot(action, robot_set_speed_func, max_speed):
    from mainController import ROBOT_ACTIONS
    
    if action == ROBOT_ACTIONS["FORWARD"]:
        robot_set_speed_func(max_speed, max_speed)
    elif action == ROBOT_ACTIONS["LEFT"]:
        robot_set_speed_func(-max_speed, max_speed)
    elif action == ROBOT_ACTIONS["RIGHT"]:
        robot_set_speed_func(max_speed, -max_speed)
    elif action == ROBOT_ACTIONS["STOP"]:
        robot_set_speed_func(0.0, 0.0)