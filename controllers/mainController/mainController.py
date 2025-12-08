"""mainController controller."""

from controller import Robot, Motor, Compass, GPS, Keyboard
import sys, os, random

# Allow imports from ../lib/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import localisation
import reinforcement_learning as rl
import odometry  
import rl_integration   
import numpy as np
import copy
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

MAX_SPEED = 10.0
# Constants
TIME_STEP = 16
TARGET_POINTS_SIZE = 13
DISTANCE_TOLERANCE = 1.5
TURN_COEFFICIENT = 4.0

# Enums/Constants
X, Y, Z, ALPHA = 0, 1, 2, 3
LEFT, RIGHT = 0, 1

autopilot = True
robot = Robot()
timestep = 512

actions = []
motors = []
position_sensors = []
distance_sensors = []
touch_sensor = None
state = RobotState()

odom = None


def robot_set_speed(left, right):
    for i in range(4):
        motors[i + 0].setVelocity(left)
        motors[i + 4].setVelocity(right)


class Vector:
    def __init__(self, u, v):
        self.u = float(u)
        self.v = float(v)

targets = [
    Vector(-2.762667, -23.877770)
]
current_target_index = 0
autopilot = True
old_autopilot = True
old_key = -1

def modulus_double(a, m):
    return math.fmod(a, m) if a >= 0 else math.fmod(a, m) + m

def norm(v):
    return math.sqrt(v.u * v.u + v.v * v.v)

def normalize(v):
    n = norm(v)
    if n != 0:
        v.u /= n
        v.v /= n

def minus(v1, v2):
    return Vector(v1.u - v2.u, v1.v - v2.v)

def robot_set_speed(left_speed, right_speed):
    for i in range(4):
        motors[i].setVelocity(left_speed)
        motors[i + 4].setVelocity(right_speed)

def check_keyboard():
    global autopilot, old_autopilot, old_key

    speeds = [0.0, 0.0]
    key = robot.keyboard.getKey()

    if key != -1:
        key_char = chr(key) if key >= 0 else key

        if key == Keyboard.UP:
            speeds[LEFT] = MAX_SPEED
            speeds[RIGHT] = MAX_SPEED
            autopilot = False
        elif key == Keyboard.DOWN:
            speeds[LEFT] = -MAX_SPEED
            speeds[RIGHT] = -MAX_SPEED
            autopilot = False
        elif key == Keyboard.RIGHT:
            speeds[LEFT] = MAX_SPEED
            speeds[RIGHT] = -MAX_SPEED
            autopilot = False
        elif key == Keyboard.LEFT:
            speeds[LEFT] = -MAX_SPEED
            speeds[RIGHT] = MAX_SPEED
            autopilot = False
        elif key_char == 'P':
            if key != old_key:
                position_3d = gps.getValues()
                print(f"position: {{{position_3d[X]:.6f}, {position_3d[Y]:.6f}}}")
        elif key_char == 'A':
            if key != old_key:
                autopilot = not autopilot

    if autopilot != old_autopilot:
        old_autopilot = autopilot
        if autopilot:
            print("auto control")
        else:
            print("manual control")

    robot_set_speed(speeds[LEFT], speeds[RIGHT])
    old_key = key

def run_autopilot():
    global current_target_index

    speeds = [0.0, 0.0]

    position_3d = gps.getValues()
    north_3d = compass.getValues()

    position = Vector(position_3d[X], position_3d[Y])
    target = targets[current_target_index]

    direction = minus(target, position)
    distance = norm(direction)
    normalize(direction)

    robot_angle = math.atan2(north_3d[0], north_3d[1])
    target_angle = math.atan2(direction.v, direction.u)
    beta = modulus_double(target_angle - robot_angle, 2.0 * math.pi) - math.pi

    if beta > 0:
        beta = math.pi - beta
    else:
        beta = -beta - math.pi

    if distance < DISTANCE_TOLERANCE:
        current_target_index += 1
        current_target_index %= TARGET_POINTS_SIZE
        
        suffix = "th"
        if current_target_index == 1:
            suffix = "st"
        elif current_target_index == 2:
            suffix = "nd"
        elif current_target_index == 3:
            suffix = "rd"
            
        print(f"{current_target_index}{suffix} target reached")
        
    else:
        base_speed = MAX_SPEED - math.pi 
        speeds[LEFT] = base_speed + TURN_COEFFICIENT * beta
        speeds[RIGHT] = base_speed - TURN_COEFFICIENT * beta
    print("Front 1 is ", distance_sensors["ds_front1"].getValue())
    if ((distance_sensors["ds_front1"].getValue() < 200)):
        print("Collision imineinfe")
        robot_set_speed(MAX_SPEED, -MAX_SPEED)
    else:
        robot_set_speed(speeds[LEFT], speeds[RIGHT])
    

def get_action(state):
    index = 0
    heading = rl.heading_to_index(state.heading)
    print("Heading is ", heading)
    if ((distance_sensors["ds_right1"].getValue() < 50 and distance_sensors["ds_left1"].getValue() < 50) or (heading == 3 and distance_sensors["ds_front1"].getValue() > 100)):
        index = 0
    if ((distance_sensors["ds_front1"].getValue() < 100 or heading == 2 ) and distance_sensors["ds_left1"].getValue() > 50):
        index = 1
    if ((distance_sensors["ds_front1"].getValue() < 100 or heading == 0 ) and distance_sensors["ds_right1"].getValue() > 50):
        index = 2
    else:
        index = random.randint(1,2)
    print("Index is ", index)
    print("DistanceSensor right1 ", distance_sensors["ds_right1"].getValue())
    print("DistanceSensor front1 ", distance_sensors["ds_front1"].getValue())
    print("DistanceSensor left1 ", distance_sensors["ds_left1"].getValue())
    rl_integration.execute_action_on_robot(index, robot_set_speed, MAX_SPEED)
    return index


def check_collision():
    for sensor in distance_sensors:
        if (distance_sensors[sensor].getValue() < 1):
            return True


def check_cargo():
    if (touch_sensor.getValue() == 0):
        return False


def main():
    global state
    global distance_sensors
    global touch_sensor
    global odom
    global compass
    global gps

    names = [
        "left motor 1", "left motor 2", "left motor 3", "left motor 4",
        "right motor 1", "right motor 2", "right motor 3", "right motor 4",
    ]

    for name in names:
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)

    for motor in motors:
        possition_sensor = motor.getPositionSensor()
        possition_sensor.enable(timestep)
        position_sensors.append(possition_sensor)

    touch_sensor = robot.getDevice("touch_sensor")
    touch_sensor.enable(timestep)
    distance_sensors = localisation.init_distance_sensors(robot, timestep)

    gps = robot.getDevice("gps")
    compass = robot.getDevice("compass")
    gps.enable(timestep)
    compass.enable(timestep)
    robot.keyboard.enable(timestep)
    odom = odometry.Odometry()
    left_positions_init = [ps.getValue() for ps in position_sensors[:4]]
    right_positions_init = [ps.getValue() for ps in position_sensors[4:]]
    odom.start_pos(left_positions_init, right_positions_init)

    previous_action = None
    previous_state = None
    count = 0
    has_collided = False
    cargo = True

    while robot.step(timestep) != -1:

        left_positions = [ps.getValue() for ps in position_sensors[:4]]
        right_positions = [ps.getValue() for ps in position_sensors[4:]]
        odom.update(left_positions, right_positions)

        sensor_readings = localisation.read_sensors(distance_sensors)
        particles = odom.get_particles()
        localisation.update_particle_weights(particles, sensor_readings)
        particles = localisation.low_variance_resample(particles)
        odom.particles = particles

        check_cargo()
        
        state = rl_integration.get_current_state_from_localization(gps, distance_sensors, odom)

        #if (state.position_x + 24 > 50 or state.position_x + 24 < 0 or state.position_y + 24 > 50 or state.position_y + 24 < 0):
            #break

        if (previous_action != None and previous_state != None):
            has_collided = check_collision()
            cargo = check_cargo()
        check_keyboard()
        run_autopilot()
        #if (has_collided or cargo == False):
            #print("Collision detected")
            #break

        #previous_state = copy.deepcopy(state)
        #previous_action = get_action(state)

    print("Q_table svaed")

if __name__ == "__main__":
    main()
    
    
    
