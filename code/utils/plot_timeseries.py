import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import os

def plot_timeseries(x : np.ndarray, 
                    y : np.ndarray,
                    heading : np.ndarray, 
                    speed : np.ndarray,
                    SAMP_FREQ : Union[int, float],
                    num_confs : int = 0,
                    IDs : np.ndarray = None,
                    plot_mean : bool = False,
                    description : str = None,
                    saved : bool = True,
                    output_folder : str = 'output/trajectory/',
                    output_file : str = 'leadership_dynamics',
                    output_format : str = 'png'):
    '''
    Plot trajectories + time series of heading and speed in a given time window.

    Parameters
    -----
    x, y : numpy array of float
        Shape (num_datapoints, N).
        x & y positions.
    heading, speed : numpy array of float
        Shape (num_datapoints, N).
    SAMP_FREQ : int or float
        sample frequency (in Hz).

    [Optional]
    num_confs (default = 0) : int
        * if 0 -> each pedestrian has a different color
        * if not 0 -> the first [num_confs] pedestrians are plotted red, and 
        otherwise black
    IDs (default = None) : numpy array
        list of pedestrian IDs
    plot_mean (default = False) : bool
        * if True -> mean time series of all pedestrains are plotted.
    description (default = None) : str
        * if not None -> added to the figure title
    saved (default = True) : bool
        * if True -> save plot
        * if False -> show plot
    output_folder : str
        output path
    output_file : str
        output file name
    output_format : str
        output file format (e.g., 'png', 'svg')

    Returns
    -----
    No returning value.
    '''
    num_datapoints, N = x.shape
    time = np.arange(num_datapoints)/SAMP_FREQ

    title_fontsize = 18
    label_fontsize = 15

    title = "Trajectories"
    if description is not None:
        title += f" ({description})"
    if num_confs!=0:
        title += "\nred: confederates, black: participants"
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 12))

    # Plot trajectories
    gs = axs[0,0].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:, 0]:
        ax.remove()
    ax1 = fig.add_subplot(gs[0:, 0])
    # ax1.plot(x, y, linewidth=2)

    colors = []
    if IDs is None:
        IDs = np.arange(1,N+1)
    if num_confs==0:
        for i in range(N):
            color = ax1.plot(x[:, i], y[:, i], linewidth=2)[0].get_color()
            colors.append(color)
    else:
        for i, ID in enumerate(IDs):
            color = ax1.plot(x[:, i], y[:, i], linewidth=2, 
                             color=("red" if (ID <= num_confs) else "black")
                             )[0].get_color()
            colors.append(color)
    for i, ID in enumerate(IDs):
        # circle & ID at the position in the first frame where the value is not NaN
        # only if the pedestrian is not missing the entire time
        if ~np.isnan(x[:,i]).all(): 
            ind = np.where(~np.isnan(x[:,i]))[0][0]
            ax1.plot(x[ind, i], y[ind, i], 
                     marker='o', markersize=15, color=colors[i])
            ax1.text(x[ind, i], y[ind, i], str(ID), 
                     color='white', fontsize=12, 
                     horizontalalignment='center', verticalalignment='center')
    ax1.grid(True)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$x$ [m]', fontsize=label_fontsize)
    ax1.set_ylabel('$y$ [m]', fontsize=label_fontsize)
    ax1.set_title(title, fontsize=title_fontsize)

    # plot heading & speed time-series on the right
    ax2 = plt.subplot(2, 2, 2)  # heading
    if num_confs==0:
        ax2.plot(time, heading, linewidth=2)
    else:
        for i in range(N):
            ax2.plot(time, heading[:, i], 
                     linewidth=2, color=("red" if (i < num_confs) else "black"))
    if plot_mean:
        ax2.plot(time, np.mean(heading, axis=1), color='greenyellow', linewidth=2)
    ax2.grid(True)
    ax2.set_xlim(0, max(time))
    # ax2.set_ylim(-180, 180)
    ax2.set_ylim(-10, 20)
    ax2.set_xlabel('time [s]', fontsize=label_fontsize)
    ax2.set_ylabel('heading [°]', fontsize=label_fontsize)
    ax2.set_title('Time series of heading', fontsize=title_fontsize)

    ax3 = plt.subplot(2, 2, 4)  # speed
    if num_confs==0:
        ax3.plot(time, speed, linewidth=2)
    else:
        for i in range(N):
            ax3.plot(time, speed[:, i], linewidth=2, color=("red" if (i < num_confs) else "black"))
    if plot_mean:
        ax3.plot(time, np.mean(speed, axis=1), color='greenyellow', linewidth=2)
    ax3.grid(True)
    ax3.set_xlim(0, max(time))
    ax3.set_ylim(0, np.ceil(np.nanmax(speed)))
    ax3.set_xlabel('time [s]', fontsize=label_fontsize)
    ax3.set_ylabel('speed [m/s]', fontsize=label_fontsize)
    ax3.set_title('Time series of speed', fontsize=title_fontsize)

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

