"""mainController controller."""

from controller import Robot, Motor
import sys, os, random

# Allow imports from ../lib/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import localisation
import reinforcement_learning as rl
import odometry  
import rl_integration   
import numpy as np
import copy

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


def run_autopilot():
    robot_set_speed(MAX_SPEED, MAX_SPEED)


def get_action():
    index = rl.q_value_action(state)
    rl_integration.execute_action_on_robot(index, robot_set_speed, MAX_SPEED)
    return index


def check_collision():
    for sensor in distance_sensors:
        if (distance_sensors[sensor].getValue() < 50):
            return True


def check_cargo():
    if (touch_sensor.getValue() == 0):
        return False


def main():
    global state
    global distance_sensors
    global touch_sensor
    global odom

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

        if (state.position_x + 24 > 50 or state.position_x + 24 < 0 or
                state.position_y + 24 > 50 or state.position_y + 24 < 0):
            break

        if (previous_action != None and previous_state != None):
            has_collided = check_collision()
            cargo = check_cargo()
            rl.q_value_update(previous_state, state, previous_action, has_collided, cargo)

        if (has_collided or cargo == False):
            break

        previous_state = copy.deepcopy(state)
        previous_action = get_action()

    rl.save_q_table("../lib/q_table")
    print("Q_table svaed")


if __name__ == "__main__":
    main()