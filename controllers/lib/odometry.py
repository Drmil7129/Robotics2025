import math
import random

PI = math.pi

increments_per_tour = 1000.0
axis_wheel_ratio = 0.03416
wheel_diameter_left = 0.0416
wheel_diameter_right = 0.0404
scaling_factor = 1

class Particle:
    def __init__(self, x=0.0, y=0.0, theta=0.0, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class ProbabilisticMotionModel:
    def __init__(self):
      
        self.alpha1 = 0.1  # Rotation error due to rotation
        self.alpha2 = 0.05 # Rotation error due to translation
        self.alpha3 = 0.1  # Translation error due to translation
        self.alpha4 = 0.05 # Translation error due to rotation

    def sample_gaussian(self, variance):
        if variance <= 0: return 0
        return random.gauss(0, math.sqrt(variance))

    def prediction_step(self, particles, delta_dist, delta_theta):
   
        for p in particles:
    
            rot_noise_variance = (self.alpha1 * abs(delta_theta) + 
                                  self.alpha2 * abs(delta_dist))
            
            trans_noise_variance = (self.alpha3 * abs(delta_dist) + 
                                    self.alpha4 * abs(delta_theta))

            sampled_rot = delta_theta + self.sample_gaussian(rot_noise_variance)
            sampled_dist = delta_dist + self.sample_gaussian(trans_noise_variance)

  
            p.x += sampled_dist * math.cos(p.theta + sampled_rot / 2.0)
            p.y += sampled_dist * math.sin(p.theta + sampled_rot / 2.0)
            p.theta += sampled_rot

            if p.theta > PI:
                p.theta -= 2 * PI
            elif p.theta < -PI:
                p.theta += 2 * PI
        
        return particles

class OdometryConfiguration:
    def __init__(self):
        self.wheel_distance = (
            axis_wheel_ratio
            * scaling_factor
            * (wheel_diameter_right + wheel_diameter_left)
            / 2
        )
        self.wheel_conversion_left = (
            wheel_diameter_left * scaling_factor * PI / increments_per_tour
        )
        self.wheel_conversion_right = (
            wheel_diameter_right * scaling_factor * PI / increments_per_tour
        )
    
class OdometryState:
    def __init__(self):
        self.pos_left_prev = []
        self.pos_right_prev = []

class OdometryResult:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


class Odometry:
    def __init__(self):
        self.config = OdometryConfiguration()
        self.state = OdometryState()
        self.result = OdometryResult()

        self.motion_model = ProbabilisticMotionModel()

        self.num_particles = 100
        self.particles = [
            Particle(0.0, 0.0, 0.0, 1.0 / self.num_particles) 
            for _ in range(self.num_particles)
        ]

    def start_pos(self, pos_left, pos_right):
        self.result.x = 0.0
        self.result.y = 0.0
        self.result.theta = 0.0

        self.state.pos_left_prev = list(pos_left)
        self.state.pos_right_prev = list(pos_right)

        self.particles = [
            Particle(0.0, 0.0, 0.0, 1.0 / self.num_particles) 
            for _ in range(self.num_particles)
        ]

    def update(self, pos_left_list, pos_right_list):

        if len(pos_left_list) != len(self.state.pos_left_prev) or \
           len(pos_right_list) != len(self.state.pos_right_prev):
            raise ValueError("Number of wheels provided does not match start_pos configuration")

        deltas_pos_right = []
        for i, current_pos in enumerate(pos_right_list):
            delta = current_pos - self.state.pos_right_prev[i]
            deltas_pos_right.append(delta)

    
        deltas_pos_left = []
        for i, current_pos in enumerate(pos_left_list):
            delta = current_pos - self.state.pos_left_prev[i]
            deltas_pos_left.append(delta)

        avg_delta_pos_left = sum(deltas_pos_left) / len(deltas_pos_left)
        avg_delta_pos_right = sum(deltas_pos_right) / len(deltas_pos_right)
    

        delta_left = avg_delta_pos_left * self.config.wheel_conversion_left
        delta_right = avg_delta_pos_right * self.config.wheel_conversion_right
        delta_dist = (delta_right + delta_left) / 2.0
  


        delta_theta = (delta_right - delta_left) / self.config.wheel_distance
        theta2 = self.result.theta + delta_theta / 2.0

        delta_x = delta_dist * math.cos(theta2)
        delta_y = delta_dist * math.sin(theta2)

        self.result.x += delta_x
        self.result.y += delta_y
        self.result.theta += delta_theta
        print(f"Odometry Update: Δx={delta_x}, Δy={delta_y}, Δθ={self.result.theta }")
      


        self.particles = self.motion_model.prediction_step(self.particles, delta_dist, delta_theta)    

        self.state.pos_left_prev = list(pos_left_list)
        self.state.pos_right_prev = list(pos_right_list)

