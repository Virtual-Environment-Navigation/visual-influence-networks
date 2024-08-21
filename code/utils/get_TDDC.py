# get TDDC values
# Nov 14, 2023 KY

import numpy as np
import matplotlib.pyplot as plt
import os
from numpy.typing import NDArray
from typing import Any, Union

def get_TDDC_ij(xvel_i : NDArray[Any], 
                yvel_i : NDArray[Any],
                xvel_j : NDArray[Any],
                yvel_j : NDArray[Any],
                SAMP_FREQ : Union[int, float], 
                TAU : Union[int, float] = 3,
                OMEGA : Union[int, float] = 40
                ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    '''
    Compute TDDC for a given pair (i, j). Used in get_TDDC().

    Parameters
    -----
    xvel_i : numpy array
        Shape (num_frames,). Represents velocities on x axis for pedestrian i.
    yvel_i : numpy array
        Shape (num_frames,). Represents velocities on y axis for pedestrian i.
    xvel_j : numpy array
        Shape (num_frames,). Represents velocities on x axis for pedestrian j.
    yvel_j : numpy array
        Shape (num_frames,). Represents velocities on y axis for pedestrian j.
    SAMP_FREQ : int or float
        sampling rate (in Hz)

    [Optional]
    TAU (default = 3) : int or float
        TAU window size (in s)
    OMEGA (default = 40) : int or float
        the number of samples on the left and on the right when getting TDDC
    
    Returns
    -----
    h_ij : numpy array 
        the scaler product with shape (num_TAU, num_frames).
    TDDC_ij : numpy array 
        TDDC values with shape (num_TAU, num_frames).
    index_ij : numpy array 
        max tau with shape (num_frames,). Missing data is indicated by NaNs.
    '''
    # ----- init -----
    num_frames = len(xvel_i)    # number of frames (= time points)
    num_TAU = 2 * (TAU * SAMP_FREQ) + 1     # number of TAU

    # ===== calculate the scaler product (indicator of alignment) =====
    # h_ij : represents the scalar product between the heading direction of 
    # pedestrian i at time t and that of pedestrian j at delayed time t+τ.
    # --- get velocities for i at t ---
    vel_i = np.column_stack((xvel_i, yvel_i))   # (num_frames, 2)
    # --- get velocities for j at t+τ ---
    TAU_index = (np.tile(np.arange(num_frames), (num_TAU, 1)) 
                 - TAU * SAMP_FREQ
                 + (num_TAU - np.tile(np.arange(num_TAU), (num_frames, 1)).T)
                 - 1)   # adjust for python indexing. Shape : (num_TAU, num_frames)
    TAU_index = np.clip(TAU_index, 0, num_frames-1)     # limit TAU_index to [0, num_frames-1]
    vel_j_TAU = np.stack([xvel_j[TAU_index], yvel_j[TAU_index]], axis=2) # (num_TAU, num_frames, 2)
    # --- get the scaler product vel_ij ---
    norm_vel_i = np.linalg.norm(vel_i, axis=1, keepdims=True)
    norm_vel_j_TAU = np.linalg.norm(vel_j_TAU, axis=2, keepdims=True)
    h_ij = np.einsum('jk,ijk->ij', vel_i/norm_vel_i, vel_j_TAU/norm_vel_j_TAU)  # (num_TAU, num_frames)
    ''' ----- einsum does the same as the following ----- '''
    # h_ij = np.zeros((num_TAU, num_frames))
    # for k in range(num_frames):     # for each t [0, num_frames-1]
    #     h_ij[:, k] = np.array([ np.dot(vel_i[k,:] / norm_vel_i[k,:],
    #                                      vel_j_TAU[h,k,:] / norm_vel_j_TAU[h,k,:])
    #                               for h in range(num_TAU-1, -1, -1) ])
    # h_ij[:, k] = np.flip(h_ij, axis=0)
    

    # ===== calculate TDDC =====
    # TDDC: C_ij(t,τ) = h_ij[(t-ω*Δt):(t+ω*Δt)+1,τ] / (2ω+1)
    # https://www.nature.com/articles/s41598-020-75551-2
    # --- get k range for each frame ---
    # These are not the actual values of k, but indices of h_ij.
    # Because these values are indices, we don't have to multiply by sample frequency.
    # for a given frame: low_index = (t-ω*Δt), high_index = (t+ω*Δt)
    low_ind = np.maximum(np.arange(num_frames) - OMEGA, 0).astype(int)    # (num_frames,)
    high_ind = np.minimum(np.arange(num_frames) + OMEGA, num_frames - 1).astype(int)   # (num_frames,)
    # --- get TDDC for a given i & j ---
    # sum of h_ij[(t-ω*Δt):(t+ω*Δt)+1, τ] = cumulative sum of h_ij[:(t+ω*Δt)+1, τ] 
    #                                       - cumulative sum of h_ij[:(t-ω*Δt), τ]
    cumsum_h_ij = np.nancumsum(h_ij[:, :], axis=1)  # ignore nan; (num_TAU, num_frames)
    TDDC_ij = (cumsum_h_ij[:, high_ind] - cumsum_h_ij[:, low_ind-1]) \
          / (high_ind - low_ind + 1)   # (num_TAU, num_frames)
    TDDC_ij[:, low_ind == 0] = cumsum_h_ij[:, high_ind[low_ind == 0]] / (high_ind[low_ind == 0] + 1)
    # --- add nan for where the velocities were missing & TDDC should be nan ---
    mask = np.isnan(h_ij)
    nan_indices = np.where(mask)
    for idx in range(len(nan_indices[0])):
        row, col = nan_indices[0][idx], nan_indices[1][idx]
        start_col = max(col - OMEGA, 0)
        end_col = min(col + OMEGA, num_frames - 1)
        mask[row, start_col:(end_col+1)] = True
    TDDC_ij = np.where(mask, np.nan, TDDC_ij)
    ''' ----- above code using cumsum does the same as the following ----- '''
    # TDDC_ij = np.zeros((num_TAU, num_frames))
    # for h in range(num_TAU - 1, -1, -1):
    #     for k in range(num_frames):
    #         TDDC_ij[h, k] = np.nansum(h_ij[h, low_indices[k]:high_indices[k] + 1])
    #         TDDC_ij[h, k] /= (high_indices[k] - low_indices[k] + 1)

    # ===== get index of max tau  =====
    index_ij = np.argmax(np.flipud(TDDC_ij), axis=0)    # (num_frames,)
    # set nan for missing data to distinguish with zeros
    index_ij[np.any(np.isnan(TDDC_ij), axis=0)] = 0     # (num_frames,)

    return h_ij, TDDC_ij, index_ij


def get_TDDC(xvel : NDArray[Any], 
             yvel : NDArray[Any],
             SAMP_FREQ : Union[int, float],
             TAU : Union[int, float] = 3,
             OMEGA : Union[int, float] = 40
             ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    '''
    Compute TDDC for all pairs (N x N) in a given time window, using 
    get_TDDC_ij().

    Parameters
    -----
    xvel : numpy array
        Shape (num_frames, N). Represents velocities on x axis for all 
        pedestrians in a given time window.
    yvel : numpy array
        Shape (num_frames, N). Represents velocities on y axis for all 
        pedestrians in a given time window.
    SAMP_FREQ : int or float
        sampling rate (in Hz).

    [Optional]
    TAU (default = 3) : int or float
        TAU window size (in s)
    OMEGA (default = 40) : int or float
        the number of samples on the left and on the right when getting TDDC

    Returns
    ----
    h : numpy array
        Shape (N, N, num_TAU, num_frames). The scaler product vel_ij for 
        calculating TDDC.
    TDDC : numpy array
        Shape (N, N, num_TAU, num_frames). TDDC values. Each of the 4 axes 
        indicates:
        (1) agent i (the pedestrian who are "leading")
        (2) agent j (the pedestrian who are "following")
        (3) num_TAU = number of time points on y axis in TDDC heat maps
        (4) num_frames = number of time points on x axis in TDDC heat maps
    index : numpy array
        Shape (N, N, num_frames). Represents indices of max tau.
    '''
    # init
    N = xvel.shape[1]
    num_frames = xvel.shape[0]    # number of frames (= time points)
    num_TAU = 2 * (TAU * SAMP_FREQ) + 1     # number of TAU
    h = np.zeros((N, N, num_TAU, num_frames))       # the scaler product (h)
    TDDC = np.zeros((N, N, num_TAU, num_frames))    # TDDC (c)
    index = np.zeros((N, N, num_frames))            # index for max(TAU)

    for i in range(N):
        for j in range(N):
            if i != j:
                h[i,j,:,:], TDDC[i,j,:,:], index[i,j,:] = get_TDDC_ij(
                    xvel[:, i], yvel[:, i], xvel[:, j], yvel[:, j], 
                    SAMP_FREQ, TAU, OMEGA)

    return h, TDDC, index


def plot_TDDC_ij(TDDC_ij : NDArray[Any], 
                 index_ij : NDArray[Any], 
                 SAMP_FREQ : Union[int, float],
                 TAU : Union[int, float] = 3,
                 CLIM_INF : Union[int, float] = 0, 
                 CLIM_SUP : Union[int, float] = 1,
                 title : str = None,
                 saved : bool = True,
                 output_folder  : str = 'output/TDDC/',
                 output_file : str = 'TDDC',
                 output_format : str = 'png') -> None :
    '''
    Plot a TDDC heat map for a given pair (i, j) in a given time window.

    Parameters
    -----
    TDDC_ij : numpy array
        Shape (num_tau, num_frames). TDDC values for a given pair.
    index_ij : numpy array
        Shape (num_frames). Indices of max tau for a given pair.
    SAMP_FREQ : int or float
        sampling rate (in Hz).
        
    [Optional]
    TAU (default = 3) : int or float
        TAU window size (in s)
    CLIM_INF (default = 0): int or float
        the lower end of the data range that the colormap covers.
    CLIM_SUP (default = 1): int or float
        the upper end of the data range that the colormap covers.
    title (default = None) : str
        figure title
    saved (default = True) : bool
        whether to save the plot or to show
    output_folder (default='output/TDDC/') : str
        where to save the folder. used only if saved is True.
    output_file (default='TDDC') : str
        used only if saved is True.
    output_format (default='png') : str
        used only if saved is True.
    
    Returns
    -----
    No returning value.
    '''
    num_tau, num_frames = TDDC_ij.shape
    fig, ax = plt.subplots()

    # fill the heat map with values
    contour = ax.contourf(np.flipud(TDDC_ij),
                          levels=50,
                          vmin=CLIM_INF, vmax=CLIM_SUP,
                          cmap="jet")
    # create color bar
    cbar = fig.colorbar(contour, ax=ax, extend='both')
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.ax.set_ylim(CLIM_INF, CLIM_SUP)    # limit range [0,1]

    # add black lines 
    ax.contour(np.flipud(TDDC_ij), 
                levels=np.arange(0.1,0.95,0.2), 
                colors='black', linewidths=0.5)
    # add white dashed lines at 0.95 level
    ax.contour(np.flipud(TDDC_ij), 
                levels=[0.95], colors='white', 
                linestyles='dashed', linewidths=1)
    # add a horizontal, black dashed line at tau = 0
    ax.axhline( y=(num_tau-1)/2+1 , 
                color='black', linestyle='dashed')
    # add a white line for max tau
    ax.plot(index_ij, color='white', linewidth=2.5)   #lime

    # add X & Y ticks
    ax.set_xticks(np.linspace(1, np.fix(num_frames), 3))
    ax.set_xticklabels([0, round(num_frames/SAMP_FREQ/2, 2), 
                        round(num_frames/SAMP_FREQ, 2)])
    ax.set_xlabel('$t$ [s]')
    ax.set_yticks(np.linspace(1, num_tau, len(range(-TAU, 4, 3))))
    ax.set_yticklabels(range(-TAU, 4, 3))
    ax.set_ylabel('$\\tau$ [s]')
            
    # add figure title
    if (title is not None):
        ax.title(title)

    # --- save ---
    if saved:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        fig.savefig(output_folder+output_file+'.'+output_format)
    else:
        plt.show()
    plt.clf()
    plt.close()
    return
