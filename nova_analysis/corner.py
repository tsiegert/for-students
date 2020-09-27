from .compute_sigma_level import compute_sigma_level
from betagen import betagen
import matplotlib.pyplot as plt
import numpy as np

wine='#8F2727'

def bayescorner(params, param_names, truth=None, color_base=wine,figsize=(10,10), colors=[], levels=[0.3,0.6,0.9]):
    """FIXME! briefly describe function

    :param params: 
    :param param_names: 
    :param truth: 
    :param color_base: 
    :returns: 
    :rtype: 

    """


    if not colors:

    
        dark, dark_highlight, mid, mid_highlight, light, light_highlight = betagen(color_base)


    else:

        assert len(colors) == 3, 'must have three colors'

        dark = colors[0]
        dark_highlight = colors[0]

        mid = colors[1]
        mid_highlight = colors[1]

        light = colors[2]
        light_highlight = colors[2]
        
    n_params = len(params)
    assert n_params == len(param_names)

    fig, axes = plt.subplots(n_params, n_params + 1, figsize=figsize, sharex=False)
    for col in range(n_params + 1):
        for row in range(n_params):

            ax = axes[row, col]
            if row > 0:
                if col < row:

                    #ax.scatter(params[row],params[col], c=dark_highlight, s=5,alpha=.05)
                    if truth is not None:
                        ax.axvline(truth[col - 1], color='grey', lw=.5)
                        ax.axhline(truth[row - 1], color='grey', lw=.5)

                    xbins, ybins, sigma = compute_sigma_level(
                        params[col - 1],
                        params[row - 1],
                        xmin=min(params[col - 1]),
                        xmax=(max(params[col - 1])),
                        ymin=min(params[row - 1]),
                        ymax=(max(params[row - 1])),
                        nbins=100)

                    ax.contourf(xbins, ybins, sigma, levels=levels, colors=(dark, mid, light), alpha=.9)
                    cc = ax.contour(
                        xbins,
                        ybins,
                        sigma,
                        levels=levels,
                        colors=(dark_highlight, mid_highlight, light_highlight),
                        linewidths=1)

                    ax.set_xlim(min(params[col - 1]), (max(params[col - 1])))

                elif col != row:

                    ax.set_visible(False)

            elif col != row:
                ax.set_visible(False)

            if row == col:

                bins = np.linspace(min(params[col - 1]), (max(params[col - 1])), 30)
                ax.hist(params[col - 1], bins=bins, fc=mid, ec=mid_highlight, lw=.5)

                #                 ax.spines['top'].set_visible(False)
                #                 ax.spines['right'].set_visible(False)
                #                 ax.spines['left'].set_visible(False)

                ax.set_xlim(min(params[col - 1]), (max(params[col - 1])))

            if col != 0:
                ax.set_yticks([])

            elif row > 0:
                ax.set_ylabel(param_names[row - 1])

            if row != len(params) - 1:
                ax.set_xticks([])

            else:
                ax.set_xlabel(param_names[col - 1])

            if (col == 0) and (row == 0):

                ax.set_yticks([])

    fig.subplots_adjust(hspace=.2, wspace=.2)

    return fig
