def polar_angle_from_altitude(y):
    # Calculate the polar angle theta from the altitude y
    theta = math.acos(y)
    return theta


def point_on_the_unit_circle(y):
    x = math.sqrt(1 - pow(y, 2))
    return {
        "point": [x, y],
        "reflected_point": [-x, y]
    }

def point_on_circle(y, radius):
    x = math.sqrt(radius - pow(y, 2))
    return {
        "point": [x, y],
        "reflected_point": [-x, y]
    }

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]






