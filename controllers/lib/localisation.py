# localisation.py
import math
import random  # for resampling

# ================== MAP / SENSOR ASSUMPTIONS ==================

SENSOR_MAX_RANGE_M = 10.0
SENSOR_MAX_RAW = 1000.0

# --- Map in odometry frame ---
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

    # solid(6) wall
    ((16.58, 19.67), (16.58, 14.67)),

    # solid(2) wall
    ((25.15, 13.84), (28.68, 10.31)),

    # solid(4) wall
    ((37.44, 24.18), (37.44, 29.18)),

     # solid(3)
    ((24.63, 42.24), (28.16, 38.71)),

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


    max_range = SENSOR_MAX_RAW
    d = min(distance, max_range) / max_range  # normalise to [0, 1]
    return math.exp(-d)


# ================== RAYCAST GEOMETRY WITH WALL + OBSTACLE SEGMENTS ==================

def ray_segment_intersection(ray_origin, ray_dir, p1, p2):
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
    dir_x = math.cos(beam_theta)
    dir_y = math.sin(beam_theta)

    closest = max_range  # default if no hit

    for (p1, p2) in ALL_SEGMENTS:
        t = ray_segment_intersection(origin, (dir_x, dir_y), p1, p2)
        if t is not None and 0.0 <= t < closest:
            closest = t

    return closest


def expected_distance_for_sensor(sensor_name, x, y, theta,
                                 max_range=SENSOR_MAX_RANGE_M):
    cfg = SENSOR_CONFIG.get(sensor_name, None)
    if cfg is None:
        return max_range

    dtheta = cfg["angle"]
    dx = cfg["dx"]
    dy = cfg["dy"]

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

    return math.exp(-(error ** 2) / (2 * sigma ** 2))


# ================== PER-SENSOR LIKELIHOOD ==================

def compute_single_sensor_likelihood(sensor_name, readings, x, y, theta, sigma):
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
                                   sigma_all=100.0,
                                   eps=1e-9):
    
    likelihoods = []
    
    for sname in SENSOR_NAMES:
        # Now uses the single parameter: sigma_all
        res = compute_single_sensor_likelihood(sname, readings, x, y, theta, sigma_all)
        
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
        for p in particles:
            p.weight /= total_w
    else:
        n = len(particles)
        if n > 0:
            uniform = 1.0 / n
            for p in particles:
                p.weight = uniform


# ================== LOW-VARIANCE RESAMPLING ==================

def low_variance_resample(particles):
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
