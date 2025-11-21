"""mainController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

from controller import Robot,Motor
import math
import numpy as np
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import lib.reinforcement_learning as rl

class RobotState:
    def __init__(self, position_x=0.0, position_y=0.0, heading=0.0, has_cargo=True):
        self.position_x = position_x
        self.position_y = position_y
        self.heading = heading
        self.has_cargo = has_cargo
        
    def __repr__(self):
        return f"RobotState(x={self.position_x}, y={self.position_y}, heading={self.heading}, cargo={self.has_cargo})"

ROBOT_ACTIONS = {
    'FORWARD': 0,
    'LEFT': 1,
    'RIGHT': 2,
    'STOP': 3

}

TIME_STEP = 512
TARGET_POINTS_SIZE = 13
DISTANCE_TOLERANCE = 1.5
MAX_SPEED = 4.0
TURN_COEFFICIENT = 4.0

autopilot = True
robot = Robot()
actions = []
motors = []
state = [0,0,0]
def robot_set_speed(left,right):
  for i in range(4):
       motors[i + 0].setVelocity(left)
       motors[i + 4].setVelocity(right)
       

#0 is foward, 1 is turn left, 2 is turn right, 3 is backwards
def index_to_action(index):
    if (index == 0):
        robot_set_speed(MAX_SPEED, MAX_SPEED)
    elif(index ==1):
        robot_set_speed( -1 * MAX_SPEED, MAX_SPEED)
    elif(index ==2):
        robot_set_speed( MAX_SPEED, -1 *  MAX_SPEED)
    elif(index ==3):
        robot_set_speed( 0,0)


def run_autopilot():
    speeds = [0.0, 0.0]
    speeds[0] = MAX_SPEED
    speeds[1] = MAX_SPEED
    robot_set_speed(speeds[0], speeds[1])

def get_action():
    index = rl.q_value_action(state)
    index_to_action(index)
    return index
    
def update_state():
    global state
    index = random.randint(0,2)
    
    state[index] += 1
    if (index == 2 and state[index] > 3 or index != 2 and state[index] > 49):
        state[index] = 0
    
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep()) 



def main():
    names = ["left motor 1",  "left motor 2",  "left motor 3",  "left motor 4",
             "right motor 1", "right motor 2", "right motor 3", "right motor 4"]
    for name in names:
        motor = robot.getDevice(name)
        motor.setPosition(float('inf'))
        motors.append(motor)
    
    previous_state = [0,0,0]
    previous_action = None
    while robot.step(TIME_STEP) != -1:
        if (previous_action != None and previous_state != None):
            rl.q_value_update(previous_state,state,previous_action)
        previous_state = state.copy()
        previous_action = get_action()
        update_state()
        
if __name__ == "__main__":
    main()