# Copyright (C) 2022 - Simleek <simulatorleek@gmail.com> - MIT License

import numpy as np


def ls_ellipsoid(cloud):
    # from: https://stackoverflow.com/a/58532308

    # finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # linear least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two

    # change xx from vector of length N to Nx1 matrix, so we can use hstack
    cloud = np.asarray(cloud)
    xx = cloud[:, 0]
    yy = cloud[:, 1]
    zz = cloud[:, 2]
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    #  math: Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    k = np.ones_like(x)  # column of ones
    j = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))

    m = np.linalg.lstsq(j, k, rcond=None)
    sol = m[0]

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    sol = np.append(sol, -1)

    return sol
