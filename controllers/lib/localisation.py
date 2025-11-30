# localisation.py
import math
import random  # for resampling

# ---- Map / sensor assumptions ----
WALL_X = 10.0              
WALL_Y_LEFT = -40.0       
WALL_Y_RIGHT = -50.0       

SENSOR_MAX_RANGE_M = 10.0 
SENSOR_MAX_RAW = 1000.0   


SENSOR_NAMES = [
    "ds_front1", "ds_front2", "ds_front3",
    "ds_left1", "ds_left2",
    "ds_right1", "ds_right2",
    "ds_back1", "ds_back2", "ds_back3",
]


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


# ================== GEOMETRY: EXPECTED DISTANCES (POSE-BASED) ==================
# All of these are purely from pose (x, y, theta).
# Odometry provides (x, y, theta) for each particle, so no GPS is used here.

def expected_front_distance_from_wall_m_pose(x, y, theta):
    dist = WALL_X - x
    if dist < 0:
        dist = 0.0
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_left_distance_from_wall_m_pose(x, y, theta):
    """
    Expected distance from robot to left wall, given pose.
    """
    dist = WALL_Y_LEFT - y
    if dist < 0:
        dist = 0.0
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_right_distance_from_wall_m_pose(x, y, theta):
    """
    Expected distance from robot to right wall, given pose.
    """
    dist = y - WALL_Y_RIGHT
    if dist < 0:
        dist = 0.0
    return min(dist, SENSOR_MAX_RANGE_M)


# ================== RAW / METERS CONVERSION + GAUSSIAN ==================

def distance_m_to_raw(dist_m):
    d = max(0.0, min(dist_m, SENSOR_MAX_RANGE_M))
    return (d / SENSOR_MAX_RANGE_M) * SENSOR_MAX_RAW


def gaussian_likelihood(error, sigma):
    """
    Likelihood of seeing this error under N(0, sigma^2)
    """
    return math.exp(-(error ** 2) / (2 * sigma ** 2))


# ================== SENSOR GROUPING HELPERS ==================

def _average_of_keys(readings, keys):
    vals = [readings[k] for k in keys if k in readings]
    if not vals:
        return None
    return sum(vals) / len(vals)


def get_front_reading(readings):
    # average of the three front sensors
    return _average_of_keys(readings, ["ds_front1", "ds_front2", "ds_front3"])


def get_left_reading(readings):
    # average of the left sensors
    return _average_of_keys(readings, ["ds_left1", "ds_left2"])


def get_right_reading(readings):
    # average of the right sensors
    return _average_of_keys(readings, ["ds_right1", "ds_right2"])


# ================== POSE-BASED MEASUREMENT MODELS ==================

def compute_front_likelihood_from_pose(readings, x, y, theta, sigma=100.0):
    """
    Front sensor likelihood given a pose (x, y, theta).
    Uses the *average* of the front sensors.
    """
    z_front = get_front_reading(readings)
    if z_front is None:
        return None

    dist_expected_m = expected_front_distance_from_wall_m_pose(x, y, theta)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_front - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_front,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


def compute_left_likelihood_from_pose(readings, x, y, theta, sigma=100.0):
    """
    Left sensor likelihood given a pose (x, y, theta).
    Uses the *average* of the left sensors.
    """
    z_left = get_left_reading(readings)
    if z_left is None:
        return None

    dist_expected_m = expected_left_distance_from_wall_m_pose(x, y, theta)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_left - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_left,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


def compute_right_likelihood_from_pose(readings, x, y, theta, sigma=100.0):
    """
    Right sensor likelihood given a pose (x, y, theta).
    Uses the *average* of the right sensors.
    """
    z_right = get_right_reading(readings)
    if z_right is None:
        return None

    dist_expected_m = expected_right_distance_from_wall_m_pose(x, y, theta)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_right - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_right,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


# ================== COMBINED SENSOR WEIGHT ==================

def combined_weight(front_info, left_info, right_info, eps=1e-9):
    """
    Multiply the three sensor likelihoods together, with clamping so
    we never hit exactly zero.
    """
    w = 1.0
    for info in (front_info, left_info, right_info):
        if info is None:
            continue
        lik = info.get("likelihood", None)
        if lik is None:
            continue
        # clamp to [eps, 1.0] to avoid zeroing everything
        lik = max(min(lik, 1.0), eps)
        w *= lik
    return w


def compute_sensor_weight_for_pose(readings, x, y, theta,
                                   sigma_front=100.0, sigma_side=100.0,
                                   eps=1e-9):
    """
    For a single pose (x, y, theta), compute all three sensor likelihoods
    and return a combined weight.
    This is what we will use for each *particle*.
    """
    front_info = compute_front_likelihood_from_pose(
        readings, x, y, theta, sigma=sigma_front
    )
    left_info = compute_left_likelihood_from_pose(
        readings, x, y, theta, sigma=sigma_side
    )
    right_info = compute_right_likelihood_from_pose(
        readings, x, y, theta, sigma=sigma_side
    )

    w = combined_weight(front_info, left_info, right_info, eps=eps)
    return w, front_info, left_info, right_info


# ================== PARTICLE WEIGHT UPDATE (USES ODOMETRY POSES) ==================

def update_particle_weights(particles, readings,
                            sigma_front=100.0, sigma_side=100.0,
                            eps=1e-9):
    """
    Given:
      - particles: list of objects with attributes x, y, theta, weight
                   (these come from your Odometry.motion_model.prediction_step)
      - readings: current sensor readings dict

    Update each particle's weight based on the sensor model,
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

