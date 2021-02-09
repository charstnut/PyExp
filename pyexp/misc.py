import numpy as np

# The inputs should be numpy arrays
# Integration and finding derivatives are given by scipy.integrate... and scipy.misc.derivative


def cluster(pts, eps):
    '''
    Given a bunch of points, for example, with coordinates (x, y, z); and given a limit radius of clustering;
    Return coordinates array of clusters (cluster radius is the limit)
    '''
    dim = len(pts[0])  # Dimension
    coord = np.array([])

    while (len(pts) > 0):
        center = pts[0]
        cap = 0  # capping iteration
        while True:  # Equivalent to "do...while"
            cap += 1
            dist = pts - center
            dist = np.linalg.norm(
                dist, axis=1)  # Calculate distance from other points
            cluster_pts = pts[dist < eps]  # Select potential clustered points
            center_new = np.average(cluster_pts,
                                    axis=0)  # Determine new center

            if (center_new == center).all or (
                    cap > 50):  # terminate if the center point is not changing
                coord = np.hstack((coord, center_new))
                dist_new = pts - center_new
                dist_new = np.linalg.norm(dist_new, axis=1)
                pts = pts[dist_new >= eps]
                break

            center = center_new

    return coord.reshape(-1, dim)


def gen_non_lin_reg(fun, xdata, ydata, fun_ops=None, yerror=None, p0=None):
    '''
	General non-linear regression assuming xdata (a M-variable x N-sampling point array) is accurate
	ydata should be one dimensional; so is yerror
	initial values should be passed, otherwise will use default (all zeros)
	fun_ops should be a list to be passed into fun as (*fun_ops)
	'''
    pass


# Execute when called as a script
if __name__ == '__main__':
    print("This script is run when using cmd line.")
