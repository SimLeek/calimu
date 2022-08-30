# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import numpy as np
from numpy.linalg import inv


def by_zero(_):
    return 0, 0, 0


def by_bounds(cloud):
    cloud = np.asarray(cloud)

    avg_x = (np.max(cloud[:, 0]) + np.min(cloud[:, 0])) / 2
    avg_y = (np.max(cloud[:, 1]) + np.min(cloud[:, 1])) / 2
    avg_z = (np.max(cloud[:, 2]) + np.min(cloud[:, 2])) / 2

    return avg_x, avg_y, avg_z


def by_average(cloud):
    cloud = np.asarray(cloud)

    avg_x = np.average(cloud[:, 0])
    avg_y = np.average(cloud[:, 1])
    avg_z = np.average(cloud[:, 2])

    return avg_x, avg_y, avg_z


def by_sphere_fit(cloud):
    """
    by_sphere_fit: fit a given set of 3D points (x, y, z) to a sphere.
    Args:
       cloud: a two-dimensional numpy array, of which each row represents (x, y, z) coordinates of a point
    Returns:
       radius, x0, y0, z0
    """
    # thanks: https://wuyang-li1990.medium.com/point-cloud-sphere-fitting-cc619c0f7ced
    cloud = np.asarray(cloud)
    row_num = cloud.shape[0]
    a = np.ones((row_num, 4))
    a[:, 0:3] = cloud

    f = np.sum(np.multiply(cloud, cloud), axis=1)

    sol, residuals, rank, sing_val = np.linalg.lstsq(a, f, rcond=None)

    return sol[0] / 2.0, sol[1] / 2.0, sol[2] / 2.0


def by_ellipsoid_fit(poly):
    # Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    a_mat = np.array(
        [
            [2 * poly[0], poly[3], poly[4], poly[6]],
            [poly[3], 2 * poly[1], poly[5], poly[7]],
            [poly[4], poly[5], 2 * poly[2], poly[8]],
            [poly[6], poly[7], poly[8], 2 * poly[9]],
        ]
    )

    a3 = a_mat[0:3, 0:3]
    a3inv = inv(a3)
    ofs = poly[6:9]
    center = np.matmul(a3inv, -ofs)

    return center
