import numpy as np
from numpy.typing import NDArray
from numbers import Number
from typing import Literal
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def plot_leadership_dynamics(ranks : NDArray[np.integer], 
                             measure_type : Literal['NI', 'NBI', 'CI', 'CBI'], 
                             description : str = None,
                             saved : bool = True,
                             output_folder : str = 'output/leadership_dynamics/',
                             output_file : str = 'leadership_dynamics',
                             output_format : str = 'png') -> None:
    '''
    Plot changes in leadership ranking for a given measure for each pedestrian.

    Parameters
    -----
    ranks : numpy array of integers 
        Shape (num_networks, N).
        leadership ranking
    measure: a string indicating the leadership measure type
        one of: NI, NBI, CI, CBI

    [Optional]
    description (default = None) : str
        If not None, it will be included in the title.
    saved (default= True) : bool
    output_folder (default = 'output/leadership_dynamics/') : str
    output_file (default = 'leadership_dynamics') : str
    output_format (default = 'png') : str

    Returns
    -----
    No returning value.
    '''
    # init
    num_networks, N = ranks.shape
    x = np.arange(1, num_networks+1)

    measure_type_dict = {
        'NI' : 'Net influence',
        'NBI' : 'Net binary influence',
        'CI' : 'Cumulative influence',
        'CBI' : 'Cumulative binary influence',
    }
    measure_type_full = measure_type_dict[measure_type]

    title = measure_type_full + ' ranking over time'
    if description is not None:
        title += ' ' + description
    
    # --- plot ---
    _, ax = plt.subplots(figsize=(10, 6))
    for i in range(N):
        ax.scatter(x, ranks[:, i], label=f'Participant {i + 1}')
        ax.plot(x, ranks[:, i], alpha=0.35)

    ax.set_xlabel('Networks', fontsize=15)
    ax.set_ylabel('Leadership ranking', fontsize=15)
    ax.set_title(title, fontsize=18)
    ax.set_xlim(0.5, num_networks+0.5)
    ax.set_ylim(0.5, N+0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # X ticks: only integers
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.invert_yaxis()
    plt.tight_layout()

    # --- save ---
    if saved:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder+output_file+'.'+output_format)
    else:
        plt.show()
    plt.clf()
    plt.close()
    return


def plot_confidence_ellipse(x : NDArray[np.number],
                            y : NDArray[np.number],
                            ax : plt.Axes, 
                            n_std : Number = 3.0, 
                            facecolor : str = 'none', 
                            **kwargs) -> Ellipse:
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    From: 
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

    Parameters
    -----
    x, y : numpy array of floats
        Shape (N,).
        xy positions of the ellipses.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    [Optional]
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -----
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

