"""mainController controller."""

from controller import Robot, Motor
import sys, os

# Make sure we can import from ../lib/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))
import localisation   # Mohammed's localisation module


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

TIME_STEP = 16
TARGET_POINTS_SIZE = 13
DISTANCE_TOLERANCE = 1.5
MAX_SPEED = 7.0
TURN_COEFFICIENT = 4.0
REWARD_PER_DISTANCE = 1
NUM_OF_STATES = 2500
NUM_OF_ACTIONS = 6

# Only ONE Robot() instance
autopilot = True
robot = Robot()
timestep = int(robot.getBasicTimeStep())

actions = []
motors = []


def robot_set_speed(left, right):
    for i in range(4):
        motors[i + 0].setVelocity(left)
        motors[i + 4].setVelocity(right)


# 0 is forward, 1 is turn left, 2 is turn right, 3 is backwards
def index_to_action(index):
    if index == 0:
        robot_set_speed(MAX_SPEED, MAX_SPEED)
    elif index == 1:
        robot_set_speed(-MAX_SPEED, MAX_SPEED)
    elif index == 2:
        robot_set_speed(MAX_SPEED, -MAX_SPEED)
    elif index == 3:
        robot_set_speed(-MAX_SPEED, -MAX_SPEED)
    # indices 4,5 can be defined later


def run_autopilot():
    robot_set_speed(MAX_SPEED, MAX_SPEED)


def main():
    # --- initialise motors ---
    names = [
        "left motor 1",
        "left motor 2",
        "left motor 3",
        "left motor 4",
        "right motor 1",
        "right motor 2",
        "right motor 3",
        "right motor 4",
    ]
    for name in names:
        motor = robot.getDevice(name)
        motor.setPosition(float("inf"))
        motor.setVelocity(0.0)
        motors.append(motor)

    # --- DEBUG: list all devices on the robot ---
    print("=== Devices on robot ===")
    for i in range(robot.getNumberOfDevices()):
        dev = robot.getDeviceByIndex(i)
        print(f"- {dev.getName()}")
    print("========================")

    # --- Mohammed: initialise distance sensors via localisation module ---
    distance_sensors = localisation.init_distance_sensors(robot, timestep)

    # --- GPS for expected measurement model (temporarily using true pose) ---
    gps = robot.getDevice("gps")
    gps.enable(timestep)

    # --- main loop ---
    while robot.step(timestep) != -1:
        if autopilot:
            run_autopilot()
        # later RL can call index_to_action(best_action_index)

        # 1) Read raw sensors
        readings = localisation.read_sensors(distance_sensors)
        print("Readings:", readings)

        # 2) Optional simple per-sensor likelihoods (debug)
        simple_liks = localisation.compute_simple_likelihoods(readings)
        print("Simple likelihoods:", simple_liks)

        # 3) Proper p(z | x) for each sensor using map + GPS
        front_info = localisation.compute_front_likelihood(readings, gps, sigma=100.0)
        left_info = localisation.compute_left_likelihood(readings, gps, sigma=100.0)
        right_info = localisation.compute_right_likelihood(readings, gps, sigma=100.0)

        if front_info is not None:
            print(
                "[Front likelihood] "
                f"measured_raw={front_info['measured_raw']:.1f}, "
                f"expected_raw={front_info['expected_raw']:.1f}, "
                f"error={front_info['error']:.1f}, "
                f"likelihood={front_info['likelihood']:.4f}"
            )

        if left_info is not None:
            print(
                "[Left likelihood]  "
                f"measured_raw={left_info['measured_raw']:.1f}, "
                f"expected_raw={left_info['expected_raw']:.1f}, "
                f"error={left_info['error']:.1f}, "
                f"likelihood={left_info['likelihood']:.4f}"
            )

        if right_info is not None:
            print(
                "[Right likelihood] "
                f"measured_raw={right_info['measured_raw']:.1f}, "
                f"expected_raw={right_info['expected_raw']:.1f}, "
                f"error={right_info['error']:.1f}, "
                f"likelihood={right_info['likelihood']:.4f}"
            )


if __name__ == "__main__":
    main()
