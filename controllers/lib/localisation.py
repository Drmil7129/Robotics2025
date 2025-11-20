# localisation.py
import math

# ---- Map / sensor assumptions ----
WALL_X = 10.0              # front wall x-position
WALL_Y_LEFT = -40.0        # left wall y-position
WALL_Y_RIGHT = -50.0       # right wall y-position

SENSOR_MAX_RANGE_M = 10.0  # meters (matches DistanceSensor lookupTable)
SENSOR_MAX_RAW = 1000.0    # raw value at max range

# Names of the distance sensors on Moose
SENSOR_NAMES = ["ds_front", "ds_left", "ds_right"]


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

def expected_front_distance_from_wall_m_pose(x, y, theta):

    dist = WALL_X - x
    if dist < 0:
        dist = 0.0 
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_left_distance_from_wall_m_pose(x, y, theta):

    dist = WALL_Y_LEFT - y
    if dist < 0:
        dist = 0.0 
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_right_distance_from_wall_m_pose(x, y, theta):
    dist = y - WALL_Y_RIGHT
    if dist < 0:
        dist = 0.0  
    return min(dist, SENSOR_MAX_RANGE_M)


# ---------- GPS wrappers (for debugging with true pose) ----------

def expected_front_distance_from_wall_m(gps):
    x, y, z = gps.getValues()
    return expected_front_distance_from_wall_m_pose(x, y, 0.0)


def expected_left_distance_from_wall_m(gps):
    x, y, z = gps.getValues()
    return expected_left_distance_from_wall_m_pose(x, y, 0.0)


def expected_right_distance_from_wall_m(gps):
    x, y, z = gps.getValues()
    return expected_right_distance_from_wall_m_pose(x, y, 0.0)


# ================== RAW / METERS CONVERSION + GAUSSIAN ==================

def distance_m_to_raw(dist_m):
    d = max(0.0, min(dist_m, SENSOR_MAX_RANGE_M))
    return (d / SENSOR_MAX_RANGE_M) * SENSOR_MAX_RAW


def gaussian_likelihood(error, sigma):
    return math.exp(-(error ** 2) / (2 * sigma ** 2))


# ================== POSE-BASED MEASUREMENT MODELS ==================

def compute_front_likelihood_from_pose(readings, x, y, theta, sigma=100.0):
    z_front = readings.get("ds_front")
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
    z_left = readings.get("ds_left")
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
    z_right = readings.get("ds_right")
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
def combined_weight(front_info, left_info, right_info, eps=1e-9):
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
# ---------- GPS wrappers (keep your current behaviour) ----------

def compute_front_likelihood(readings, gps, sigma=100.0):
    x, y, z = gps.getValues()
    return compute_front_likelihood_from_pose(readings, x, y, 0.0, sigma)


def compute_left_likelihood(readings, gps, sigma=100.0):
    x, y, z = gps.getValues()
    return compute_left_likelihood_from_pose(readings, x, y, 0.0, sigma)


def compute_right_likelihood(readings, gps, sigma=100.0):
    x, y, z = gps.getValues()
    return compute_right_likelihood_from_pose(readings, x, y, 0.0, sigma)


# ================== COMBINED SENSOR WEIGHT ==================

def combined_weight(front_info, left_info, right_info):
    w = 1.0
    if front_info is not None:
        w *= front_info["likelihood"]
    if left_info is not None:
        w *= left_info["likelihood"]
    if right_info is not None:
        w *= right_info["likelihood"]
    return w
