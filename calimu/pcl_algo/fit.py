# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import numpy as np
import sklearn
from numpy.linalg import eig, inv
import sklearn.decomposition


def from_axis_aligned_bounding_box(cloud, center):
    # followers Kris Winer's method of fitting, which is much simpler if the data is reliable:
    # https://github.com/kriswiner/MPU6050/wiki/Simple-and-Effective-Magnetometer-Calibration
    # should be much faster than other methods, but much less accurate.

    cloud = np.asarray(cloud)

    avg_delta_x = (np.max(cloud[:, 0]) - np.min(cloud[:, 0])) / 2
    avg_delta_y = (np.max(cloud[:, 1]) - np.min(cloud[:, 1])) / 2
    avg_delta_z = (np.max(cloud[:, 2]) - np.min(cloud[:, 2])) / 2

    t = np.eye(4)
    t[0:3, 3] = center
    t[0, 0] = avg_delta_x
    t[1, 1] = avg_delta_y
    t[2, 2] = avg_delta_z
    return t, np.average([avg_delta_x, avg_delta_y, avg_delta_z])


def from_sphere(cloud, center):
    # note: as long as the center is within the sphere, the average radius calculation should be accurate
    cloud = np.asarray(cloud)
    cl = cloud.copy()

    cl[:, 0:3] = cl[:, 0:3] - center[0]

    r = np.linalg.norm(cl, axis=1)
    r_avg = np.average(r)

    t2 = np.eye(4)
    t2[0:3, 3] = center
    t2[0, 0] = r_avg
    t2[1, 1] = r_avg
    t2[2, 2] = r_avg

    return t2, r_avg


def from_pca(cloud, center):
    cloud = np.asarray(cloud)
    cl = cloud.copy()

    cl[:, 0:3] = cl[:, 0:3] - center[0:3]

    # noinspection PyUnresolvedReferences
    pca = sklearn.decomposition.PCA(n_components=3)
    _ = pca.fit_transform(cl)

    t0 = np.eye(4)
    t0[0:3, 3] = center[0:3]

    t1 = np.eye(4)
    t1[0, 0:3] = pca.components_[0]
    t1[1, 0:3] = pca.components_[1]
    t1[2, 0:3] = pca.components_[2]

    scale_vec = np.sqrt(pca.explained_variance_) * 2
    t2 = np.eye(4)
    t2[0, 0] = scale_vec[0]
    t2[1, 1] = scale_vec[1]
    t2[2, 2] = scale_vec[2]

    t3 = np.eye(4)
    t3[0, 0:3] = pca.components_[0]
    t3[1, 0:3] = pca.components_[1]
    t3[2, 0:3] = pca.components_[2]
    t3 = np.linalg.inv(t3)

    t4 = t0 @ t1 @ t2 @ t3

    return t4, np.average(scale_vec)


def from_ellipsoid(center, poly):
    a_mat = np.array(
        [
            [2 * poly[0], poly[3], poly[4], poly[6]],
            [poly[3], 2 * poly[1], poly[5], poly[7]],
            [poly[4], poly[5], 2 * poly[2], poly[8]],
            [poly[6], poly[7], poly[8], 2 * poly[9]],
        ]
    )

    # Center the ellipsoid at the origin
    t_ofs = np.eye(4)
    t_ofs[3, 0:3] = center
    r = np.dot(t_ofs, np.dot(a_mat, t_ofs.T))

    r3 = r[0:3, 0:3]
    s1 = -r[3, 3]
    r3_s = r3 / s1
    (el, ec) = eig(r3_s)

    recip = 1.0 / np.abs(el)
    radii = np.sqrt(recip)

    t0 = np.eye(4)
    t0[0:3, 3] = center[0:3]

    t1 = np.eye(4)
    t1[0:3, 0:3] = ec

    t2 = np.eye(4)
    t2[0, 0] = radii[0]
    t2[1, 1] = radii[1]
    t2[2, 2] = radii[2]

    t3 = np.eye(4)
    t3[0:3, 0:3] = inv(ec)

    t4 = t0 @ t1 @ t2 @ t3
    return t4, np.average(radii)
