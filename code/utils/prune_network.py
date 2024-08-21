# used to prune the network when creating network weights in get_network.py

import numpy as np
from typing import Union
import copy

from utils.get_visibility import get_visibility_all

def prune_with_visibility(weights : np.ndarray, 
                          start_frames : np.ndarray, 
                          end_frames : np.ndarray,
                          x : np.ndarray, 
                          y : np.ndarray, 
                          heading : np.ndarray,
                          vis_adjust : bool = False,
                          threshold : Union[int, float] = 0.15
                          ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Apply visibility pruning. Get original weights and return new weights.

    Parameters
    -----
    weights : numpy array of float
        Shape (num_networks, N, N).
        Network weights.
    start_frames, end_frames : numpy array of int
        Shape (num_networks,)
        Starting & ending frames (indices) of each network.
    x, y, heading : numpy array of float 
        Shape (num_networks, N)

    [Optional]
    vis_adjust (default = False) : bool
        * if True -> weights are adjusted based on visibility
        * if False -> weights are not adjusted (only pruning)
    threshold (default = 0.15) : int or float
        visibility pruning threshold for lower bound.
        range : [0,1]

    Returns
    -----
    new_weights: numpy array of float
        Shape (num_networks, N, N)
        New network weights.
    visibility_avg : numpy array of float
        Shape (num_networks, N, N)
        Average visibility used for pruning.
    '''
    ## --- init ---
    num_networks = weights.shape[0]
    new_weights = copy.copy(weights)

    ## --- get visibility ---
    # visibility at each time point 
    visibility_mat = get_visibility_all(x, y, heading)  # (num_timepoints, N, N)
    # average visibility for each network
    visibility_avg = np.array([np.mean(visibility_mat[start_frames[k]:end_frames[k]+1,:,:], axis=0)
                               for k in range(num_networks)])    # (k,N,N) = (ntwk,i,j)
    # transpose <-- weights are (k,i,j) but visibility is (k,j,i)
    visibility_avg = np.transpose(visibility_avg, (0, 2, 1))

    ## --- modify  weights based on visibility ---
    # if True: weights are adjusted based on visibility
    if vis_adjust:
        new_weights *= visibility_avg
    # set weights to 0 if visibility is below the threshold
    new_weights[visibility_avg<threshold] = 0
    new_weights[np.isnan(weights)] = np.nan

    ## --- error handling ---
    if np.any(new_weights < 0):
        raise ValueError('Invalid weight during visibility pruning: weights cannot be < 0.')
    if np.any(new_weights > 1):
        raise ValueError('Invalid weight during visibility pruning: weights cannot be > 1.')

    return new_weights, visibility_avg


def prune_with_timedelay(weights : np.ndarray, 
                         start_frames : np.ndarray, 
                         end_frames : np.ndarray,
                         index : np.ndarray, 
                         SAMP_FREQ : Union[int, float],
                         TAU : Union[int, float] = 3,
                         low_bound : float = 0.3
                         ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Apply time-delay pruning. Get original weights and return new weights.

    Parameters
    -----
    weights : numpy array of float
        Shape (num_networks, N, N).
        Network weights.
    start_frames, end_frames : numpy array of int
        Shape (num_networks,)
        Starting & ending frames (indices) of each network.
    index: numpy array of int
        Shape (N, N, num_frames) = (i, j, frame)
        TDDC indices.
    SAMP_FREQ : int or float
        sample frequency (in Hz)

    [Optional]
    TAU (default = 3) : int or float
        TAU window size (in s)
    low_bound (default = 0.15): int or float
        visibility pruning threshold for lower bound
        
    
    Returns
    -----
    new_weights: numpy array of float
        Shape (num_networks, N, N)
        New network weights.
    visibility_avg : numpy array of float
        Shape (num_networks, N, N)
        Average visibility used for pruning.
    '''
    ## --- init ---
    num_networks = weights.shape[0]

    ## --- get time delay ---
    # get average tau (indices on the vertical axis)
    avg_timedelay_ind = np.array([np.mean(index[:,:,start_frames[k]:end_frames[k]+1], axis=2)
                                  for k in range(num_networks)])    # (k,N,N) = (ntwk,i,j)
    # get average tau (in seconds)
    avg_timedelay_t = (avg_timedelay_ind - TAU*SAMP_FREQ) / SAMP_FREQ     # (k,N,N)

    ## --- modify weights based on time delay ---
    # prune if time delay is lower than a threshold
    new_weights = np.where(avg_timedelay_t<low_bound, 0, weights)
    new_weights[np.isnan(weights)] = np.nan

    ## --- error handling ---
    if np.any(new_weights < 0):
        raise ValueError('Invalid weight during time delay pruning: weights cannot be < 0.')
    if np.any(new_weights > 1):
        raise ValueError('Invalid weight during time delay pruning: weights cannot be > 1.')
    
    return new_weights, avg_timedelay_t
