import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal
import networkx as nx
import copy

from utils.get_network import create_network

'''
Compute leadership measures:

Net Influence (NI)
    * a local measure of individual influence, taking weights into account.
Net Binary Influence (NBI)
    * a local measure of individual role in the network structure, without 
    taking weights into account.
Cumulative Influence (CI)
    * a global measure of individual influence, taking weights into account
Cumulative Binary Influence (CBI)
    * a global measure of individual role in the network structure, without 
    taking weights into account.

'''

# used in get_NI, get_CI, get_NBI, get_CBI to normalize the leadership values
def normalize_leadership(leadership : NDArray[Any], 
                         norm_method : Literal['max_leadership', 
                                               'crowd_size',
                                               'max_min_leadership'] = 'max_leadership'
                        ) -> NDArray[Any]:
    '''
    Normalize the leadership values in a given network.

    Parameters
    -----
    leadership: numpy array of leadership values 
        Shape (N,). Nodes are in the order of the pedestrian ID order (0 to N-1).
    
    [Optional]
    norm_method: str
        * if 'max_leadership' (default) -> values are normalized based on the 
        maximum leadership value in the given network
        * if 'crowd_size' -> values are normalized based on crowd size (N-1)
        * if 'max_min_leadership' -> positive values are normalized by the maximum
        value, while negative values are normalized by the absolute value of
        the minimum value

    Returns
    -----
    norm_leadership: numpy array of normalized leadership values with shape (N,)
    '''
    if (norm_method=="max_leadership"):
        norm_leadership = leadership / max(leadership)
    elif (norm_method=="crowd_size"):
        N = leadership.shape[0]
        norm_leadership = leadership / (N-1)
    elif (norm_method=="max_min_leadership"):
        norm_leadership = copy.copy(leadership).astype(np.float64)
        norm_leadership[leadership > 0] /= max(leadership)
        norm_leadership[leadership < 0] /= abs(min(leadership))
    return norm_leadership
    

def get_NI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal['max_leadership', 
                                 'crowd_size',
                                 'max_min_leadership'] = 'max_leadership'
          ) -> NDArray[Any]: 
    '''
    Compute Net Influence (NI; weighted outdegree - weighted indegree) for all
    N pedestrians in a given network.

    Parameters
    -----
    weights: numpy array of network weights
        Shape (N, N).

    [Optional]
    normalize : bool
        * if True (default) -> normalize NI
        * if False -> NI is not normalized

    order : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents 
        pedestrian j (followers).
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders).

    norm_method: str
        * if 'max_leadership' (default) -> values are normalized based on the 
        maximum leadership value in the given network
        * if 'crowd_size' -> values are normalized based on crowd size (N-1)
        * if 'max_min_leadership' -> positive values are normalized by the 
        maximum value, while negative values are normalized by the absolute 
        value of the minimum value

    Returns
    -----
    ni: numpy array of NI values for each pedestrian
        shape (N,).
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)
    
    G = create_network(weights, order=order)     # create a directed graph
    weighted_outdegs = np.array(G.out_degree(weight='weight'))[:,1]     # (N,)
    weighted_indegs = np.array(G.in_degree(weight='weight'))[:,1]       # (N,)
    ni = weighted_outdegs - weighted_indegs
    
    if normalize:
        ni = normalize_leadership(ni, norm_method)

    return ni


def get_NBI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal['max_leadership', 
                                 'crowd_size',
                                 'max_min_leadership'] = 'max_leadership'
           ) -> NDArray[Any]: 
    '''
    Compute Net Binary Influence (NBI; outdegree - indegree) for all N 
    pedestrians in a given network.

    Parameters
    -----
    weights: numpy array of network weights
        Shape (N, N).

    [Optional]
    normalize : bool
        * if True (default) -> normalize NBI
        * if False -> NI is not normalized

    order : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents pedestrian j 
        (followers)
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j 
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders)

    norm_method: str
        * if 'max_leadership' (default) -> values are normalized based on the 
        maximum leadership value in the given network
        * if 'crowd_size' -> values are normalized based on crowd size (N-1)
        * if 'max_min_leadership' -> positive values are normalized by the maximum
        value, while negative values are normalized by the absolute value of
        the minimum value
    
    Returns
    -----
    nbi: numpy array of NBI values for each pedestrian
        shape (N,).
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)
    
    G = create_network(weights, order=order)     # create a directed graph
    outdegs = np.array(G.out_degree())[:,1]     # (N,)
    indegs = np.array(G.in_degree())[:,1]       # (N,)
    nbi = outdegs - indegs

    if normalize:
        nbi = normalize_leadership(nbi, norm_method)
    
    return nbi


def get_CI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal['max_leadership', 
                                 'crowd_size',
                                 'max_min_leadership'] = 'max_leadership'
           ) -> NDArray[Any]:
    '''
    Compute Cumulative Influence (CI; the sum of multiplied weights of each
    edge in each possible path between the pedestrians) for all N pedestrians 
    in a given network.

    Parameters
    -----
    weights: numpy array of network weights
        Shape (N, N).

    [Optional]
    normalize : bool
        * if True (default) -> normalize NBI
        * if False -> NI is not normalized

    order : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents pedestrian j 
        (followers)
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j 
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders)

    norm_method: str
        * if 'max_leadership' (default) -> values are normalized based on the 
        maximum leadership value in the given network
        * if 'crowd_size' -> values are normalized based on crowd size (N-1)
        * if 'max_min_leadership' -> positive values are normalized by the maximum
        value, while negative values are normalized by the absolute value of
        the minimum value

    Returns
    -----
    ci: numpy array of CI values for each pedestrian
        shape (N,)
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)

    # create a directed graph
    G = create_network(weights, order=order)
    N = weights.shape[0]      # get N

    if order=="ij":
        weights = np.transpose(weights)

    # --- get paths ---
    w = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (i!=j and nx.has_path(G, source=i, target=j)):
                w[i,j] = np.sum([ np.prod([weights[p[l+1], p[l]] for l in range(len(p)-1)])
                                 for p in nx.all_simple_paths(G, source=i, target=j) ])
    ci = np.sum(w, axis=1)

    if normalize:
        ci = normalize_leadership(ci, norm_method)

    return ci


def get_CBI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal['max_leadership', 
                                 'crowd_size',
                                 'max_min_leadership'] = 'max_leadership'
           ) -> NDArray[Any]: 
    '''
    Compute Cumulative Binary Influence (CBI; the number of all possible paths starting
    from a given pedestrian) for a given network.

    Parameters
    -----
    weights: numpy array of network weights
        Shape (N, N).

    [Optional]
    normalize : bool
        * if True (default) -> normalize NBI
        * if False -> NI is not normalized

    order : str
        Indicates the order of weights.
        * if 'ij' (default) -> the 1st dimension of weights (rows) represents 
        pedestrian i (leaders), and the 2nd dimension (columns) represents pedestrian j 
        (followers)
        * if 'ji' -> the 1st dimension of weights (rows) represents pedestrian j 
        (followers), and the 2nd dimension (columns) represents pedestrian i 
        (leaders)

    norm_method: str
        * if 'max_leadership' (default) -> values are normalized based on the 
        maximum leadership value in the given network
        * if 'crowd_size' -> values are normalized based on crowd size (N-1)
        * if 'max_min_leadership' -> positive values are normalized by the maximum
        value, while negative values are normalized by the absolute value of
        the minimum value

    Returns
    -----
    cbi: numpy array of CBI values for each pedestrian
        shape (N,)
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)

    # create a directed graph
    G = create_network(weights, order=order)
    N = weights.shape[0]      # get N

    if order=="ij":
        weights = np.transpose(weights)
    elif order!="ji":
        raise ValueError('Incorrect value for order while getting NI')

    # --- get paths ---
    cbi = np.zeros((N,))
    for i in range(N):
        for j in range(N):
            if (i!=j and nx.has_path(G, source=i, target=j)):
                cbi[i] += sum(1 for path 
                              in nx.all_simple_paths(G, source=i, target=j))

    if normalize:
        cbi = normalize_leadership(cbi, norm_method)

    return cbi


def get_rank(leadership_value : NDArray[Any]) -> NDArray[Any]:
    '''
    Compute ranks for a specified leadership measure for each of N pedestrians 
    in a given network.

    Parameters
    -----
    leadership_value: numpy array of leadership values
        Shape (N,).

    Returns
    -----
    leadership_rank: numpy array with shape (N,)
        * leadership rank for each of N pedestrians
    '''
    N = np.shape(leadership_value)[0]

    # replace NaNs with -np.inf so that they get the lowest ranking
    leadership_value = np.where(np.isnan(leadership_value), 
                                -np.inf, leadership_value)

    # sort the indices in descending order (higher value = more highly ranked)
    rank_ind = np.argsort(leadership_value)[::-1][:N]
    leadership_rank = np.argsort(rank_ind) + 1
    
    return leadership_rank
