# state_transition

import math

def Predict_Next_State(current_state, action, step_size=0.5, turn_angle=15):
    from mainController import RobotState, ROBOT_ACTIONS
    
    new_x = current_state.position_x
    new_y = current_state.position_y
    new_heading = current_state.heading
    new_cargo = current_state.has_cargo
    
    if action == ROBOT_ACTIONS["FORWARD"]:
        heading_rad = math.radians(new_heading)
        new_x += step_size * math.cos(heading_rad)
        new_y += step_size * math.sin(heading_rad)
    
    elif action == ROBOT_ACTIONS["LEFT"]:
        new_heading = (new_heading + turn_angle) % 360
    
    elif action == ROBOT_ACTIONS["RIGHT"]:
        new_heading = (new_heading - turn_angle) % 360
    
    return RobotState(new_x, new_y, new_heading, new_cargo)


def Get_Transition_Probability(current_state, action, next_state):
    predicted = predict_next_state(current_state, action)
    
    tolerance = 0.01
    x_match = abs(predicted.position_x - next_state.position_x) < tolerance
    y_match = abs(predicted.position_y - next_state.position_y) < tolerance
    heading_match = abs(predicted.heading - next_state.heading) < tolerance
    
    if x_match and y_match and heading_match:
        return 1.0
    return 0.0