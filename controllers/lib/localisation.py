# localisation.py
import math
import random  # for resampling

# ================== MAP / SENSOR ASSUMPTIONS ==================

SENSOR_MAX_RANGE_M = 10.0
SENSOR_MAX_RAW = 1000.0

# --- Map in odometry frame ---
# Map size matches Odometry.initialize_global(49, 49)
# Coordinates here are in the same frame as your particles (odom.particles)
W_LEFT   = 0.0
W_RIGHT  = 49.0
H_BOTTOM = 0.0
H_TOP    = 49.0

# Four outer walls as line segments in odometry (x, y) plane
WALL_SEGMENTS = [
    ((W_LEFT,  H_BOTTOM), (W_RIGHT, H_BOTTOM)),  # bottom wall
    ((W_LEFT,  H_TOP   ), (W_RIGHT, H_TOP   )),  # top wall
    ((W_LEFT,  H_BOTTOM), (W_LEFT,  H_TOP   )),  # left wall
    ((W_RIGHT, H_BOTTOM), (W_RIGHT, H_TOP   )),  # right wall
]

# ------------------ INTERIOR OBSTACLES ------------------

OBSTACLE_SEGMENTS = [
   # solid(1) wall
    ((12.78, 26.58), (17.78, 26.58)),

    # solid(5) wall
    ((7.26, 28.21), (7.26, 23.21)),

    # solid(6) wall: translation -8.48 -7.02 6.39, rot -1.57
    # child:        translation 0.31 0.56 -1.02, rot -1.57
    ((16.58, 19.67), (16.58, 14.67)),

    # solid(2) wall: translation 1.8 -12.6 4.59, rot -0.785
    # child:         translation 0.31 0.56 -1.02, rot -1.57
    ((25.15, 13.84), (28.68, 10.31)),

    # solid(4) wall: translation 13.5 1.87 3.98, rot +1.57
    # child:         translation 0.31 0.56 -1.02, rot -1.57
    ((37.44, 24.18), (37.44, 29.18)),

     # solid(3)
    ((24.63, 42.24), (28.16, 38.71)),

    # barrels
    #oilbarrel
    ((27.05, 23.23), (28.65, 23.23)),
    ((28.65, 23.23), (28.65, 21.63)),
    ((28.65, 21.63), (27.05, 21.63)),
    ((27.05, 21.63), (27.05, 23.23)),
    #oilbarrel(2)
    ((11.40, 34.31), (13.00, 34.31)),
    ((13.00, 34.31), (13.00, 32.71)),
    ((13.00, 32.71), (11.40, 32.71)),
    ((11.40, 32.71), (11.40, 34.31)),
]



# All segments in the world
ALL_SEGMENTS = WALL_SEGMENTS + OBSTACLE_SEGMENTS

# ------------------ SENSOR CONFIGURATION ------------------
# One entry per sensor: direction + (optional) offset in robot frame.
# dx, dy are in the *robot* coordinate frame:
#   x = forward, y = left  (typical mobile-robot convention)
#
# For now we keep dx = dy = 0 for all sensors; this assumes all sensors
# are at the robot center. You can refine these later if you want.

SENSOR_CONFIG = {
    # front centre (straight ahead)
    "ds_front1": {"angle": 0.0,      "dx": 1.55,  "dy": 0.0},
    "ds_front2": {"angle": -0.4,     "dx": 1.35,  "dy": -0.65},
    "ds_front3": {"angle": +0.4,     "dx": 1.35,  "dy": +0.65},

    # sides
    "ds_left1":  {"angle": +math.pi/2, "dx": 0.60,  "dy": 0.62},
    "ds_left2":  {"angle": +math.pi/2, "dx": -0.65, "dy": 0.62},
    "ds_right1": {"angle": -math.pi/2, "dx": 0.60,  "dy": -0.70},
    "ds_right2": {"angle": -math.pi/2, "dx": -0.66, "dy": -0.70},

    # back
    "ds_back1":  {"angle":  math.pi,   "dx": -1.53,   "dy": 0.03},
    "ds_back2":  {"angle":  math.pi-0.5, "dx": -1.46, "dy": 0.66},
    "ds_back3":  {"angle": -math.pi+0.5, "dx": -1.46, "dy": -0.64},
}

SENSOR_NAMES = list(SENSOR_CONFIG.keys())


# ================== DEVICE INITIALISATION / READING ==================

def init_distance_sensors(robot, timestep):
    distance_sensors = {}
    for sname in SENSOR_NAMES:
        dev = robot.getDevice(sname)
        if dev is None:
            print(f"[WARN] Sensor '{sname}' not found on robot!")
        else:
            dev.enable(timestep)
            distance_sensors[sname] = dev
            print(f"[OK] Initialised sensor '{sname}'")
    return distance_sensors


def read_sensors(distance_sensors):
    readings = {}
    for name, sensor in distance_sensors.items():
        readings[name] = sensor.getValue()
    return readings


# ================== SIMPLE (DEBUG) LIKELIHOODS ==================

def simple_likelihood(distance: float) -> float:
    if distance <= 0:
        return 0.0

    max_range = SENSOR_MAX_RAW
    d = min(distance, max_range) / max_range  # normalise to [0, 1]
    return math.exp(-d)


def compute_simple_likelihoods(readings):
    return {name: simple_likelihood(val) for name, val in readings.items()}


# ================== RAYCAST GEOMETRY WITH WALL + OBSTACLE SEGMENTS ==================

def ray_segment_intersection(ray_origin, ray_dir, p1, p2):
    """
    Intersection between a ray and a segment.

    ray_origin: (x0, y0)
    ray_dir:    (dx, dy)   -- direction (need not be normalised)
    p1, p2:     endpoints of segment

    Returns distance t >= 0 along ray if intersection exists, else None.
    """
    x0, y0 = ray_origin
    dx, dy = ray_dir
    x1, y1 = p1
    x2, y2 = p2

    sx = x2 - x1
    sy = y2 - y1

    denom = dx * (-sy) - dy * (-sx)
    if abs(denom) < 1e-9:
        # Ray and segment are parallel or nearly so
        return None

    inv_denom = 1.0 / denom
    rx = x1 - x0
    ry = y1 - y0

    # Solve for t (along ray) and u (along segment)
    t = (rx * (-sy) - ry * (-sx)) * inv_denom
    u = (dx * ry - dy * rx) * inv_denom

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def _cast_ray_from_origin(origin, beam_theta, max_range=SENSOR_MAX_RANGE_M):
    """
    Low-level raycast from 'origin' in direction 'beam_theta'
    against ALL_SEGMENTS (outer walls + obstacles).
    """
    dir_x = math.cos(beam_theta)
    dir_y = math.sin(beam_theta)

    closest = max_range  # default if no hit

    for (p1, p2) in ALL_SEGMENTS:
        t = ray_segment_intersection(origin, (dir_x, dir_y), p1, p2)
        if t is not None and 0.0 <= t < closest:
            closest = t

    return closest


def expected_distance_generic(x, y, theta, sensor_angle,
                              max_range=SENSOR_MAX_RANGE_M):
    """
    OLD-style helper (still used by some functions):
    cast a ray from robot center with orientation (theta + sensor_angle).
    """
    beam_theta = theta + sensor_angle
    origin = (x, y)
    return _cast_ray_from_origin(origin, beam_theta, max_range)


def expected_distance_for_sensor(sensor_name, x, y, theta,
                                 max_range=SENSOR_MAX_RANGE_M):
    """
    NEW: full per-sensor beam model.

    - Uses per-sensor angle from SENSOR_CONFIG.
    - If you set dx, dy in SENSOR_CONFIG, it will also use the
      proper sensor position on the robot body.
    """
    cfg = SENSOR_CONFIG.get(sensor_name, None)
    if cfg is None:
        # Unknown sensor: just return max range
        return max_range

    dtheta = cfg["angle"]
    dx = cfg["dx"]
    dy = cfg["dy"]

    # Transform sensor offset from robot frame to world/map frame
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    sx = x + cos_t * dx - sin_t * dy
    sy = y + sin_t * dx + cos_t * dy

    beam_theta = theta + dtheta
    origin = (sx, sy)

    return _cast_ray_from_origin(origin, beam_theta, max_range)


# ================== RAW / METERS CONVERSION + GAUSSIAN ==================

def distance_m_to_raw(dist_m):
    d = max(0.0, min(dist_m, SENSOR_MAX_RANGE_M))
    return (d / SENSOR_MAX_RANGE_M) * SENSOR_MAX_RAW


def gaussian_likelihood(error, sigma):
    """
    Likelihood of seeing this error under N(0, sigma^2)
    """
    return math.exp(-(error ** 2) / (2 * sigma ** 2))


# ================== PER-SENSOR LIKELIHOOD ==================

def compute_single_sensor_likelihood(sensor_name, readings, x, y, theta, sigma):
    """
    Compute likelihood for ONE distance sensor at pose (x, y, theta).
    """
    if sensor_name not in readings:
        return None

    z_meas = readings[sensor_name]

    dist_expected_m = expected_distance_for_sensor(sensor_name, x, y, theta)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_meas - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return lik, z_meas, z_expected_raw, error


# ================== PARTICLE WEIGHT: ALL SENSORS ==================

def compute_sensor_weight_for_pose(readings, x, y, theta,
                                   sigma_front=100.0, sigma_side=100.0,
                                   eps=1e-9):
    
    likelihoods = []
    
    for sname in SENSOR_NAMES:
        res = compute_single_sensor_likelihood(sname, readings, x, y, theta, sigma_front)
        
        if res is not None:
            lik = res[0]
            # Ensure likelihood is never exactly 0 to avoid log(0) error
            likelihoods.append(max(lik, eps))

    # 2. Compute Geometric Mean
    if not likelihoods:
        return eps, None, None, None

    # Sum of logs = Log of product
    sum_log_likelihood = sum(math.log(p) for p in likelihoods)
    
    # Average the logs (this is the 1/N step)
    avg_log_likelihood = sum_log_likelihood / len(likelihoods)
    
    # Convert back to normal probability
    w = math.exp(avg_log_likelihood)

    return w, None, None, None


def update_particle_weights(particles, readings,
                            sigma_front=100.0, sigma_side=100.0,
                            eps=1e-9):
    """
    Given:
      - particles: list of objects with attributes x, y, theta, weight
      - readings: current sensor readings dict

    Update each particle's weight based on *all* distance sensors,
    then normalise so weights sum to 1.0.

    If all weights would be zero (numerical problem), we reset
    to uniform weights.
    """
    total_w = 0.0

    for p in particles:
        w, _, _, _ = compute_sensor_weight_for_pose(
            readings, p.x, p.y, p.theta,
            sigma_front=sigma_front,
            sigma_side=sigma_side,
            eps=eps,
        )
        p.weight = w
        total_w += w

    if total_w > 0.0:
        # Normalise
        for p in particles:
            p.weight /= total_w
    else:
        # Fallback: uniform weights
        n = len(particles)
        if n > 0:
            uniform = 1.0 / n
            for p in particles:
                p.weight = uniform


# ================== LOW-VARIANCE RESAMPLING ==================

def low_variance_resample(particles):
    """
    Standard low-variance (systematic) resampling.

    - Input: list of particles with .x, .y, .theta, .weight
    - Output: new list of particles (same length), approximately resampled
      according to weights, with equal weights 1/N.

    We assume weights are already normalised.
    """
    N = len(particles)
    if N == 0:
        return []

    step = 1.0 / N
    r = random.random() * step
    c = particles[0].weight
    i = 0

    new_particles = []
    ParticleClass = type(particles[0])

    for m in range(N):
        U = r + m * step
        while U > c and i < N - 1:
            i += 1
            c += particles[i].weight

        # Copy pose of chosen particle, reset weight to uniform
        src = particles[i]
        new_p = ParticleClass(src.x, src.y, src.theta, 1.0 / N)
        new_particles.append(new_p)

    return new_particles


# ================== POSE ESTIMATION FROM PARTICLES ==================

def estimate_pose(particles):
    """
    Estimate (x, y, theta) from a list of particles.
    Simple mean for x,y and circular mean for theta.
    """
    n = len(particles)
    if n == 0:
        return 0.0, 0.0, 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_sin = 0.0
    sum_cos = 0.0

    for p in particles:
        sum_x += p.x
        sum_y += p.y
        sum_sin += math.sin(p.theta)
        sum_cos += math.cos(p.theta)

    x_hat = sum_x / n
    y_hat = sum_y / n
    theta_hat = math.atan2(sum_sin / n, sum_cos / n)

    return x_hat, y_hat, theta_hat
