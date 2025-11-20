# localisation.py
import math

# ---- Map / sensor assumptions (Mohammed) ----
WALL_X = 10.0              # front wall x-position
WALL_Y_LEFT = -40.0        # left wall y-position
WALL_Y_RIGHT = -50.0       # right wall y-position

SENSOR_MAX_RANGE_M = 10.0  # meters (matches DistanceSensor lookupTable)
SENSOR_MAX_RAW = 1000.0    # raw value at max range

# Names of the distance sensors on Moose
SENSOR_NAMES = ["ds_front", "ds_left", "ds_right"]


def init_distance_sensors(robot, timestep):
    """
    Initialise all distance sensors defined in SENSOR_NAMES.
    Returns a dict: name -> DistanceSensor device.
    """
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
    """
    Read raw values from all distance sensors.
    Returns: dict name -> raw_value.
    """
    readings = {}
    for name, sensor in distance_sensors.items():
        readings[name] = sensor.getValue()
    return readings


# ---------- simple debugging likelihood (per sensor only) ----------

def simple_likelihood(distance: float) -> float:
    """
    Simple placeholder likelihood based only on the measured distance.
    (Still useful for debugging.)
    """
    if distance <= 0:
        return 0.0

    max_range = SENSOR_MAX_RAW
    d = min(distance, max_range) / max_range  # normalise to [0, 1]
    return math.exp(-d)


def compute_simple_likelihoods(readings):
    """
    Compute simple_likelihood for each sensor based on its raw reading.
    Returns: dict name -> likelihood.
    """
    return {name: simple_likelihood(val) for name, val in readings.items()}


# ---------- expected distances from map + GPS (true pose for now) ----------

def expected_front_distance_from_wall_m(gps):
    """
    Expected front sensor distance in METERS given the true pose from GPS,
    assuming a vertical wall at x = WALL_X.
    """
    x, y, z = gps.getValues()
    dist = WALL_X - x   # front wall is in +x direction
    if dist < 0:
        dist = 0.0  # robot has passed the wall
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_left_distance_from_wall_m(gps):
    """
    Expected LEFT sensor distance (ds_left), pointing +y, to wall at y = WALL_Y_LEFT.
    """
    x, y, z = gps.getValues()
    # Robot is at smaller y (~-45), wall at larger y (-40), so free space is wall - robot
    dist = WALL_Y_LEFT - y
    if dist < 0:
        dist = 0.0  # robot is already beyond the wall on +y side
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_right_distance_from_wall_m(gps):
    """
    Expected RIGHT sensor distance (ds_right), pointing -y, to wall at y = WALL_Y_RIGHT.
    """
    x, y, z = gps.getValues()
    # Wall is at smaller y (-50), sensor looks -y, free space is robot - wall
    dist = y - WALL_Y_RIGHT
    if dist < 0:
        dist = 0.0  # robot is beyond the wall on -y side
    return min(dist, SENSOR_MAX_RANGE_M)


# ---------- conversion + Gaussian likelihood ----------

def distance_m_to_raw(dist_m):
    """
    Convert distance in meters to the raw sensor units (0..SENSOR_MAX_RAW),
    assuming linear mapping (matches DistanceSensor lookupTable).
    """
    d = max(0.0, min(dist_m, SENSOR_MAX_RANGE_M))
    return (d / SENSOR_MAX_RANGE_M) * SENSOR_MAX_RAW


def gaussian_likelihood(error, sigma):
    """Gaussian likelihood p(z | x) ∝ exp(-(error^2)/(2σ^2))."""
    return math.exp(-(error ** 2) / (2 * sigma ** 2))


# ---------- full measurement models per sensor ----------

def compute_front_likelihood(readings, gps, sigma=100.0):
    """
    Proper measurement likelihood for the front sensor.
    """
    z_front = readings.get("ds_front")
    if z_front is None:
        return None

    dist_expected_m = expected_front_distance_from_wall_m(gps)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_front - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_front,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


def compute_left_likelihood(readings, gps, sigma=100.0):
    """
    Proper measurement likelihood for the LEFT sensor.
    """
    z_left = readings.get("ds_left")
    if z_left is None:
        return None

    dist_expected_m = expected_left_distance_from_wall_m(gps)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_left - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_left,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


def compute_right_likelihood(readings, gps, sigma=100.0):
    """
    Proper measurement likelihood for the RIGHT sensor.
    """
    z_right = readings.get("ds_right")
    if z_right is None:
        return None

    dist_expected_m = expected_right_distance_from_wall_m(gps)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_right - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_right,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }
