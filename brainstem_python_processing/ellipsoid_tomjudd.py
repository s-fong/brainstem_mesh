from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from numpy.linalg import eig, inv


def ls_ellipsoid(xx, yy, zz):
    # finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two

    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = xx[:, np.newaxis]
    y = yy[:, np.newaxis]
    z = zz[:, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))
    K = np.ones_like(x)  # column of ones

    # np.hstack performs a loop over all samples and creates
    # a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC = np.dot(InvJTJ, np.dot(JT, K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC, -1)

    return (eansa)


def polyToParams3D(vec, printMe):
    # gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)

    # Algebraic form: X.T * Amat * X --> polynomial form

    if printMe: print('\npolynomial\n', vec)

    Amat = np.array(
        [
            [vec[0], vec[3] / 2.0, vec[4] / 2.0, vec[6] / 2.0],
            [vec[3] / 2.0, vec[1], vec[5] / 2.0, vec[7] / 2.0],
            [vec[4] / 2.0, vec[5] / 2.0, vec[2], vec[8] / 2.0],
            [vec[6] / 2.0, vec[7] / 2.0, vec[8] / 2.0, vec[9]]
        ])

    if printMe: print('\nAlgebraic form of polynomial\n', Amat)

    # See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3, 0:3]
    A3inv = inv(A3)
    ofs = vec[6:9] / 2.0
    center = -np.dot(A3inv, ofs)
    if printMe: print('\nCenter at:', center)

    # Center the ellipsoid at the origin
    Tofs = np.eye(4)
    Tofs[3, 0:3] = center
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))
    if printMe: print('\nAlgebraic form translated to center\n', R, '\n')

    R3 = R[0:3, 0:3]
    R3test = R3 / R3[0, 0]
    # print('normed \n',R3test)
    s1 = -R[3, 3]
    R3S = R3 / s1
    (el, ec) = eig(R3S)

    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)
    if printMe: print('\nAxes are\n', axes, '\n')

    inve = inv(ec)  # inverse is actually the transpose here
    if printMe: print('\nRotation matrix\n', inve)
    return (center, axes, inve, R)


def checkSolution(radii, R, data):
    # Check solution
    # Convert to unit sphere centered at origin
    #  1) Subtract off center
    #  2) Rotate points so bulges are aligned with axes (no xy,xz,yz terms)
    #  3) Scale the points by the inverse of the axes gains
    #  4) Back rotate
    # Rotations and gains are collected into single transformation matrix M

    # subtract the offset so ellipsoid is centered at origin
    xin = data[:,0]
    yin = data[:,1]
    zin = data[:,2]
    xc = xin - center[0]
    yc = yin - center[1]
    zc = zin - center[2]

    # create transformation matrix
    L = np.diag([1 / radii[0], 1 / radii[1], 1 / radii[2]])
    M = np.dot(R.T, np.dot(L, R))
    print('\nTransformation Matrix\n', M)

    # apply the transformation matrix
    [xm, ym, zm] = np.dot(M, [xc, yc, zc])
    # Calculate distance from origin for each point (ideal = 1.0)
    rm = np.sqrt(xm * xm + ym * ym + zm * zm)

    print("\nAverage Radius  %10.4f (truth is 1.0)" % (np.mean(rm)))
    print("Stdev of Radius %10.4f\n " % (np.std(rm)))


# https://github.com/minillinim/ellipsoid
def ellipsoid_plot(center, radii, rotation, ax, plot_axes=False, cage_color='b', cage_alpha=0.2):
    """Plot an ellipsoid"""

    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)

    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center

    if plot_axes:
        # make some purdy axes
        axes = np.array([[radii[0], 0.0, 0.0],
                         [0.0, radii[1], 0.0],
                         [0.0, 0.0, radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)

        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cage_color)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color=cage_color, alpha=cage_alpha)




if __name__ == '__main__':
    N = 50
    rg = [[-120, 120], [-60, 60], [-30, 30]]
    randn = np.random.rand(N, 3)
    surface = np.array([[(abs(rg[j][0]) + abs(rg[j][1])) * row[j] + rg[j][0] for j in range(3)] for row in randn])
    # get convex hull

    hullV = ConvexHull(surface)
    lH = len(hullV.vertices)
    hull = np.zeros((lH, 3))
    for i in range(len(hullV.vertices)):
        hull[i] = surface[hullV.vertices[i]]
    hull = np.transpose(hull)

    # fit ellipsoid on convex hull
    eansa = ls_ellipsoid(hull[0], hull[1], hull[2])  # get ellipsoid polynomial coefficients
    print("coefficients:", eansa)
    center, axes, inve, R = polyToParams3D(eansa, False)  # get ellipsoid 3D parameters
    print("center:", center)
    print("axes:", axes)
    print("rotationMatrix:", inve)

    # for row in surface:
    #     checkSolution(axes, inve, row)
    checkSolution(axes, inve, surface)