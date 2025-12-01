# rl_integration.py

from mainController import RobotState, ROBOT_ACTIONS, robot_set_speed, MAX_SPEED
import localisation
import reinforcement_learning as rl
import math

def get_current_state_from_localization(gps, distance_sensors, odom):
    gps_values = gps.getValues()
    x = gps_values[0]
    y = gps_values[1]
    
    heading_rad = odom.result.theta
    heading = math.degrees(heading_rad)
    
    has_cargo = True
    
    return RobotState(x, y, heading, has_cargo)


def execute_action_on_robot(action):
    if action == ROBOT_ACTIONS["FORWARD"]:
        robot_set_speed(MAX_SPEED, MAX_SPEED)
    elif action == ROBOT_ACTIONS["LEFT"]:
        robot_set_speed(-MAX_SPEED, MAX_SPEED)
    elif action == ROBOT_ACTIONS["RIGHT"]:
        robot_set_speed(MAX_SPEED, -MAX_SPEED)
    elif action == ROBOT_ACTIONS["STOP"]:
        robot_set_speed(0.0, 0.0)


def rl_step(gps, distance_sensors, odom, previous_state, has_collided, cargo_status):
    current_state = get_current_state_from_localization(gps, distance_sensors, odom)
    
    action = rl.q_value_action(current_state)
    
    execute_action_on_robot(action)
    
    if previous_state is not None:
        rl.q_value_update(previous_state, current_state, action, has_collided, cargo_status)
    
    return current_state