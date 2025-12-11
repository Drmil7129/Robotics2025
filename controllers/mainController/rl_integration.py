# rl_integration.py
# Integration layer connecting localization system to reinforcement learning

import localisation
import reinforcement_learning as rl
import math

# RobotState class - represents the robot's state in the MDP
class RobotState:
    def __init__(self, position_x=0.0, position_y=0.0, heading=0.0, has_cargo=True):
        self.position_x = position_x  # Robot's x-coordinate
        self.position_y = position_y  # Robot's y-coordinate
        self.heading = heading        # Robot's heading in degrees (0-360)
        self.has_cargo = has_cargo    # Whether robot still has cargo

    def __repr__(self):
        # String representation for debugging
        return (
            f"RobotState(x={self.position_x}, y={self.position_y}, "
            f"heading={self.heading}, cargo={self.has_cargo})"
        )

# Action space - maps action names to indices used by Q-learning
ROBOT_ACTIONS = {
    "FORWARD": 0,
    "LEFT": 1,
    "RIGHT": 2,
    "BACK": 3,
}

def get_current_state_from_localization(gps, distance_sensors, odom):
    """
    Extract current robot state from localization system.
    Gets position from MCL particles and heading from odometry.
    """
    # Get MCL particle estimate
    particles = odom.get_particles()
    x, y, theta_rad = localisation.estimate_pose(particles)
    
    # Debug output
    print(f"DEBUG MCL: x={x:.2f}, y={y:.2f}, theta={theta_rad:.2f}")
    
    # Check for invalid MCL values (NaN)
    if math.isnan(x) or math.isnan(y) or math.isnan(theta_rad):
        print("DEBUG: Using odometry fallback")
        # Fall back to odometry if MCL fails
        x = odom.result.x if not math.isnan(odom.result.x) else 0.0
        y = odom.result.y if not math.isnan(odom.result.y) else 0.0
        theta_rad = odom.result.theta if not math.isnan(odom.result.theta) else 0.0
    
    # Convert heading from radians to degrees
    heading = math.degrees(theta_rad)
    if heading < 0:
        heading += 360  # Normalize to [0, 360)
    
    # Cargo status (currently always True, updated externally)
    has_cargo = True
    
    return RobotState(x, y, heading, has_cargo)


def execute_action_on_robot(action, robot_set_speed_func, max_speed):
    """
    Execute the selected RL action by setting motor speeds.
    Translates action index to left/right wheel speed commands.
    """
    if action == ROBOT_ACTIONS["FORWARD"]:
        # Both wheels forward at max speed
        robot_set_speed_func(max_speed, max_speed)
    elif action == ROBOT_ACTIONS["LEFT"]:
        # Left wheels backward, right wheels forward (spin left)
        robot_set_speed_func(-max_speed, max_speed)
    elif action == ROBOT_ACTIONS["RIGHT"]:
        # Left wheels forward, right wheels backward (spin right)
        robot_set_speed_func(max_speed, -max_speed)
    elif action == ROBOT_ACTIONS["BACK"]:
        # Both wheels backward
        robot_set_speed_func(-max_speed, -max_speed)