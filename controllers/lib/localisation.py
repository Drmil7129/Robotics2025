# localisation.py
import math

# ---- Map / sensor assumptions (Mohammed) ----
# Front wall in +X direction
WALL_X = 10.0               # x-position of the front wall

# Side walls in Y
WALL_Y_LEFT = -40.0         # y-position of the LEFT wall
WALL_Y_RIGHT = -50.0        # y-position of the RIGHT wall

# Distance sensor characteristics
SENSOR_MAX_RANGE_M = 10.0   # meters (matches DistanceSensor.lookupTable)
SENSOR_MAX_RAW = 1000.0     # raw value at max range

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


# -------------------- Simple likelihood (debug) -------------------- #

def simple_likelihood(distance: float) -> float:
    """
    Simple placeholder likelihood based only on the measured distance.
    Still useful for debugging and quick sanity checks.
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


# -------------------- Geometry: expected distances -------------------- #

def expected_front_distance_from_wall_m(gps):
    """
    Expected front sensor distance in METERS given the true pose from GPS,
    assuming a vertical wall at x = WALL_X.
    """
    x, y, z = gps.getValues()
    dist = WALL_X - x
    if dist < 0.0:
        dist = 0.0  # robot has passed the wall
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_left_distance_m(gps):
    """
    Expected distance to the LEFT wall (at WALL_Y_LEFT).
    Robot is between the two walls in Y, so left wall is at higher Y (-40).
    """
    x, y, z = gps.getValues()
    # Distance from robot to left wall along +Y
    dist = y - WALL_Y_LEFT        # e.g. y = -45, WALL_Y_LEFT = -40 -> dist = -5 -> clamp to 0
    if dist < 0.0:
        dist = 0.0
    return min(dist, SENSOR_MAX_RANGE_M)


def expected_right_distance_m(gps):
    """
    Expected distance to the RIGHT wall (at WALL_Y_RIGHT).
    Right wall is at lower Y (-50).
    """
    x, y, z = gps.getValues()
    # Distance from robot to right wall along -Y
    dist = WALL_Y_RIGHT - y       # e.g. WALL_Y_RIGHT = -50, y = -45 -> dist = -5 -> clamp to 0
    if dist < 0.0:
        dist = 0.0
    return min(dist, SENSOR_MAX_RANGE_M)


# -------------------- Conversions + Gaussian model -------------------- #

def distance_m_to_raw(dist_m):
    """
    Convert distance in meters to raw sensor units (0..SENSOR_MAX_RAW),
    assuming a linear mapping (matches DistanceSensor.lookupTable).
    """
    d = max(0.0, min(dist_m, SENSOR_MAX_RANGE_M))
    return (d / SENSOR_MAX_RANGE_M) * SENSOR_MAX_RAW


def gaussian_likelihood(error, sigma):
    """Gaussian likelihood p(z | x) ∝ exp(-(error^2)/(2σ^2))."""
    return math.exp(-(error ** 2) / (2.0 * sigma ** 2))


# -------------------- Full likelihoods for each sensor -------------------- #

def compute_front_likelihood(readings, gps, sigma=100.0):
    """
    Compute a proper measurement likelihood for the FRONT sensor:

      - uses map (WALL_X) + GPS to get expected distance,
      - converts to raw units,
      - compares with measured raw,
      - returns a dict with all useful values.

    Returns None if 'ds_front' is not present.
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
    Measurement likelihood for the LEFT sensor 'ds_left'.
    Uses WALL_Y_LEFT and GPS.y.
    """
    z_left = readings.get("ds_left")
    if z_left is None:
        return None

    dist_expected_m = expected_left_distance_m(gps)
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
    Measurement likelihood for the RIGHT sensor 'ds_right'.
    Uses WALL_Y_RIGHT and GPS.y.
    """
    z_right = readings.get("ds_right")
    if z_right is None:
        return None

    dist_expected_m = expected_right_distance_m(gps)
    z_expected_raw = distance_m_to_raw(dist_expected_m)

    error = z_right - z_expected_raw
    lik = gaussian_likelihood(error, sigma)

    return {
        "measured_raw": z_right,
        "expected_raw": z_expected_raw,
        "error": error,
        "likelihood": lik,
    }


# -------------------- Optional helper: all three at once -------------------- #

def compute_all_likelihoods(readings, gps, sigma=100.0):
    """
    Convenience function to compute likelihood summaries for all three sensors.
    Returns a dict: sensor_name -> likelihood_info_dict (or None if missing).
    """
    return {
        "front": compute_front_likelihood(readings, gps, sigma),
        "left": compute_left_likelihood(readings, gps, sigma),
        "right": compute_right_likelihood(readings, gps, sigma),
    }
