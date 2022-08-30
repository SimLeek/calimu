# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import numpy as np


def get_err(cloud, xform):
    # transform the points in the cloud to a sphere of size 1 using xform and check how close we got
    xform_inv = np.linalg.inv(xform)
    cloud = np.asarray(cloud)
    b = np.ones((cloud.shape[0], cloud.shape[1] + 1))
    b[:, :-1] = cloud
    c = np.dot(b, xform_inv.T)
    c2 = c[:, :-1]
    r = np.linalg.norm(c2, axis=1)

    # fix for transforms that shrink the data to zero:
    # if the cloud goes to 0, the max err/dist is just 1
    # however, if the cloud goes to inf, the max err/dist is inf
    # here, we fix that so that it goes to inf in either direction
    rel_dist = r - np.ones_like(r)
    rel_dist[rel_dist < 0] = -1 / (rel_dist[rel_dist < 0] + 1) + 1

    rel_std = np.std(
        rel_dist + 1
    )  # avg is one, so this is already relative standard deviation
    mae = np.sum(np.sum(np.abs(rel_dist))) / r.shape[0]
    return rel_std, mae
