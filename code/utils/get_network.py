import numpy as np
from numpy.typing import NDArray
from typing import Any, Union, Literal
import networkx as nx
import matplotlib.pyplot as plt
import os

from utils.prune_network import *

'''
functions for generating our visual influence networks.
'''

def nan_missing_weights(weights : NDArray[Any], 
                        x : NDArray[Any], 
                        SAMP_FREQ : Union[int, float], 
                        ntwk_window_size : Union[int, float], 
                        TAU : Union[int, float] = 3
                        ) -> NDArray[Any]:
    '''
    If a pedestrian k was missing, all weights of the edges connected to k are
    set to NaN for the networks representing the missing time windows, as well
    as the networks within ±tau seconds.

    Parameters
    -----
    weights : numpy array of network weights in a given time window
        Shape (num_networks, N, N)
    x : numpy array of x positions 
        Shape (num_timepoints, N)
    SAMP_FREQ: int or float
        sample frequency (in Hz)
    ntwk_window_size : int or float
        network window size (in seconds)

    [Optional]
    TAU (default = 3) : int or float
        tau window size (in seconds)

    Returns
    -----
    weights: np array of new network weights in a given time window
        Shape : (num_networks, N, N)
    '''
    num_networks = weights.shape[0]
    num_timepoints = x.shape[0]
    # number of frames in each network
    num_frames_network = int(SAMP_FREQ * ntwk_window_size)
    # number of networks that represent TAU (3) seconds
    num_networks_TAU = int(TAU / ntwk_window_size)

    for network_ID in range(num_networks):  # [0, num_networks-1]
        # start/end networks to nan (within plus/minus tau seconds)
        start_ntwk = max(0, network_ID-num_networks_TAU)
        end_ntwk = min(num_networks-1, network_ID+num_networks_TAU)
        # start/end frames of the given network
        start_frame = num_frames_network * network_ID
        end_frame = (num_timepoints - 1) if network_ID==(num_networks-1) \
            else (num_frames_network * (network_ID+1) - 1) 
        missing_data_rows = np.any(np.isnan(x[start_frame:(end_frame+1), :]), axis=0)
        # replace values with nan
        weights[start_ntwk:end_ntwk+1, missing_data_rows, :] = np.nan
        weights[start_ntwk:end_ntwk+1, :, missing_data_rows] = np.nan

    return weights


def get_network_weights(index : NDArray[Any], 
                        x : NDArray[Any],
                        y : NDArray[Any], 
                        heading : NDArray[Any], 
                        SAMP_FREQ : Union[int, float], 
                        pruning : bool = True, 
                        TAU : Union[int, float] = 3, 
                        NTWK_WINDOW_SIZE : Union[int, float] = 0.5,
                        vis_adjust : bool = False,
                        visibility_threshold : Union[int, float] = 0.15,
                        timedelay_low_threshold : Union[int, float] = 0.3
                        ) -> NDArray[Any]:
    '''
    Get network weights based on max tau indices in a given time window.

    Parameters
    -----
    index: numpy array of max tau indices 
        Shape : (N, N, num_frames)
    x : numpy array of x positions
        Shape (num_timepoints, N)
    y : numpy array of y positions 
        Shape (num_timepoints, N)
    heading: numpy array of headings
        Shape (num_timepoints, N)
    SAMP_FREQ: int or float
        sample frequency (in Hz)
    
    [Optional]
    pruning (default = True) : bool
        whether weights should be pruned
    TAU (default = 3) : int or float
        TAU window size (in seconds)
    NTWK_WINDOW_SIZE (default = 0.5): int or float
        network window size (in seconds)
    vis_adjust (default = False) : bool
        if weights should be adjusted based on visibility
    visibility_threshold (default = 0.15) : float
    timedelay_low_threshold (default = 0.3) : float

    Returns
    -----
    weights : numpy array of network weights in the same time window.
        Shape : (num_networks, N, N)
    '''
    ## ===== init =====
    N, _, num_frames = index.shape

    ## ===== get the details of networks to generate =====
    # number of frames in each network
    num_frames_network = int(SAMP_FREQ * NTWK_WINDOW_SIZE)
    # number of networks in the given time interva;
    num_networks = int(num_frames // num_frames_network)
    # use the last time window as one network if >=half of the window size
    # combine with previous if too short (<half)
    num_frames_last_network = num_frames % num_frames_network
    if (num_frames_last_network >= (num_frames_network/2)):
        num_networks += 1
    # get the start & end frame IDs for each network
    start_frames = np.arange(start = 0, 
                             stop = num_frames_network * (num_networks-1) + 1, 
                             step = num_frames_network).astype(int) # (num_networks,)
    end_frames = (start_frames + num_frames_network - 1).astype(int)
    end_frames[-1] = num_frames - 1 

    ## ===== get initial network weights (before pruning) from max tau =====
    weights = np.zeros((num_networks, N, N))
    for ntwk in range(num_networks):
        ## get the number of frames in each network where i is leading j
        leading_mask = index[:, :, start_frames[ntwk]:end_frames[ntwk]+1] \
                            > TAU * SAMP_FREQ   # (N, N, num_frames_network)
        num_frames_lead = np.sum(leading_mask, axis=2)     # per pair, (N, N)
        ## get a proportion of time i is leading j in the given network
        weights[ntwk,:, :] = num_frames_lead / (end_frames[ntwk]-start_frames[ntwk]+1)

    ## ===== apply pruning (visibility & time delay) =====
    if (pruning==True):
        weights, _ = prune_with_visibility(weights, start_frames, end_frames,
                                           x, y, heading,
                                           vis_adjust=vis_adjust,
                                           threshold=visibility_threshold)
        weights, _ = prune_with_timedelay(weights, start_frames, end_frames,
                                          index, SAMP_FREQ,
                                          TAU=TAU,
                                          low_bound=timedelay_low_threshold)
        
    ## ===== change values to nan for missing values =====
    weights = nan_missing_weights(weights, x, 
                                  SAMP_FREQ, NTWK_WINDOW_SIZE, 
                                  TAU=TAU)
        
    return weights


def create_network(weights : NDArray[Any],
                   order : Literal["ij", 'ji'] = "ij") -> NDArray[Any]:
    '''
    Create a network (one snapshot).

    Parameters
    -----
    weights: numpy array of network weights
        Shape (N, N).
        *CAUTION*: weights are in the order of (j,i).

    [Optional]
    order : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents 
        pedestrian j (followers).
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders).

    Returns
    -----
    G2: networkx directed graph.
        Nodes are sorted in the pedestrian ID order (0 to N-1).
    '''
    if order=="ij":
        weights = np.transpose(weights)
    
    # --- Returns a graph from a 2D NumPy array ---
    # nodes are not sorted (added when edges are found)
    G1 = nx.from_numpy_array(weights.T, create_using=nx.DiGraph)
    # sort nodes/pedestrians in the order of 0 to N-1
    G2 = nx.DiGraph()
    G2.add_nodes_from(sorted(G1.nodes(data=True)))
    G2.add_edges_from(G1.edges(data=True))

    return G2


def plot_networks(weights : NDArray[Any],
                  positions : NDArray[Any],
                  num_confs : int = 0,
                  IDs = None,
                  order : str = "ij",
                  node_size : Union[int, float] = 300,
                  node_font_color : str = 'white',
                  saved : bool = True,
                  ncols : int = 5,
                  title : str = None,
                  subtitle : str = None,
                  output_folder : str = 'output/networks/',
                  output_file : str = 'network',
                  output_format : str = 'png') -> None:
    '''
    Plot a series of networks.

    Parameters
    -----
    weights : numpy array of network weights
        Shape (num_networks, N, N)
    positions : numpy array of positions
        Shape (num_networks, N, 2)

    [Optional]
    num_confs (default = 0) : int
        number of confederates
    IDs = None
    order (default = 'ij') : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents 
        pedestrian j (followers).
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders).
    node_size (default = 300) : int
        scalar or array with shape (num_networks, N)
    node_font_color (default = 'white') : str
        font color of nodes
    saved (default = True) : bool
        whether to save the plot or to show
    ncols (default = 5) : int
        number of columns for networks 
    title (default = None) : str
        main figure title
    subtitle (default = None) : str
        subtitle for each network
    output_folder (default='output/networks/') : str
        where to save the folder. used only if saved is True.
    output_file (default='network') : str
        used only if saved is True.
    output_format (default='png') : str
        used only if saved is True.
    
    Returns
    -----
    No returning value.
    '''
    num_networks = np.shape(weights)[0]
    nrows = (num_networks//ncols + 1) if (num_networks%5 != 0) \
        else (num_networks//ncols)
    ax_all = plt.figure(figsize=(5*ncols, 5*nrows)).subplots(nrows, ncols)

    is_title_None = True if (subtitle is None) else False
    
    for ntwk in range(num_networks):
        # --- initialize a network ---
        # create a network
        G = create_network(weights[ntwk,:,:], order)
        # remove missing pedestrian(s)
        missing = list(n for n in G.nodes()
                       if np.isnan(weights[ntwk,n,:]).all() )
        G.remove_nodes_from(missing)
        # node sizes
        ns = node_size if isinstance(node_size, int) \
            else node_size[ntwk,:][G.nodes]*1000
        # change the node labels -> +1
        mapping = {i: i+1 for i in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        # change the arrow thickness
        widths = nx.get_edge_attributes(G, 'weight')
        widths = {key: value*2 for key, value in widths.items()}
        # node positions (x, y)
        pos = {i: (positions[ntwk, i-1, 0], positions[ntwk, i-1, 1])
               for i in G.nodes()}
        if IDs is not None:
            mapping = {i: ID for i,ID in enumerate(IDs)}
            
        # --- plot ---
        ax = ax_all[ntwk%ncols] if nrows==1 else ax_all[ntwk//ncols, ntwk%ncols]
        # plot a network
        if num_confs==0:
            c = '#1f78b4'
        else:
            c = ['red' for _ in range(num_confs)]   # conf
            c += ['black' for _ in range(len(G.nodes()) - num_confs)]   # participant
        nx.draw_networkx_nodes(G, pos=pos,
                            node_color=c,
                            nodelist=G.nodes(), node_size=ns,
                            ax=ax)
        nx.draw_networkx_edges(G, pos=pos,
                               edgelist=widths.keys(), 
                               width=list(widths.values()),
                               ax=ax)
        nx.draw_networkx_labels(G, pos=pos,
                                labels=dict(zip(G.nodes(), G.nodes())),
                                font_color=node_font_color,
                                ax=ax)
        # subtitle
        if is_title_None:
            subtitle = f"{output_file}_network{ntwk+1}"
            if missing:
                subtitle += '\nmissing: ' + str(np.array(missing)+1)
        ax.set_title(subtitle)
        # set ax
        ax.set_axis_on()
        ax.set_aspect('equal')
        ax.tick_params(left=True, bottom=True, 
                       labelleft=True, labelbottom=True)
        # ax.set_xlabel('x [m]', fontsize=15)
        # ax.set_ylabel('y [m]', fontsize=15)
        if (ntwk%ncols == 0):
            ax.set_ylabel(r'Mean heading$\longrightarrow$', fontsize=15)

    # deal with empty ax cells
    if ((num_networks%ncols) != 0):   # when there are empty cells
        for n in range(num_networks, nrows*ncols):
            ax = ax_all[n%ncols] if nrows==1 else ax_all[n//ncols, n%ncols]
            ax.set_axis_off()

    # plot title
    # plt.ylabel(r'Mean heading$\longrightarrow$', fontsize=18)

    if title is not None:
        plt.suptitle(title, fontsize=20, position=(0.5,0.95))

    # --- save ---
    if saved:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + output_file + '.' + output_format)
    else:
        plt.show()
    plt.clf()
    plt.close()
    return 