"""mainController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

from controller import Robot,Motor

TIME_STEP = 16
TARGET_POINTS_SIZE = 13
DISTANCE_TOLERANCE = 1.5
MAX_SPEED = 7.0
TURN_COEFFICIENT = 4.0
autopilot = True
robot = Robot()
actions = []
motors = []

def robot_set_speed(left,right):
  for i in range(4):
       motors[i + 0].setVelocity(left)
       motors[i + 4].setVelocity(right)
       

def reward_function():
    return 0
    
def run_autopilot():
    speeds = [0.0, 0.0]
    speeds[0] = MAX_SPEED
    speeds[1] = MAX_SPEED
    robot_set_speed(speeds[0], speeds[1])
 
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

def main():
    names = ["left motor 1",  "left motor 2",  "left motor 3",  "left motor 4",
             "right motor 1", "right motor 2", "right motor 3", "right motor 4"]
    for name in names:
        motor = robot.getDevice(name)
        motor.setPosition(float('inf'))
        motors.append(motor)
        
    while robot.step(timestep) != -1:
        if (autopilot):
            run_autopilot()

if __name__ == "__main__":
    main()