import numpy as np
import scipy.stats as stats


def compute_sigma_level(trace1, trace2, nbins=20, xmin=0, xmax=np.pi, ymin=0, ymax=100, bw_method='scott'):
    """From a set of traces, bin by number of standard deviations"""

    dxy = stats.gaussian_kde(np.vstack([trace1, trace2]),bw_method=bw_method)

    xx = np.linspace(xmin, xmax, nbins)
    yy = np.linspace(ymin, ymax, nbins)

    xbins, ybins = np.meshgrid(xx, yy)

    grid_coords = np.append(xbins.reshape(-1, 1), ybins.reshape(-1, 1), axis=1)

    L = dxy(grid_coords.T)
    L = L.reshape(nbins, nbins)

    #L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)
