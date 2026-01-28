import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal
import networkx as nx
import copy

from utils.get_network import create_network

'''
Compute leadership measures:

Take weights into account:
* Direct Influence (DI): a local measure of individual influence
* Branching Influence (BI): a global measure of individual influence

Without taking weights into account (number of edges):
* Direct Binary Influence (DBI)
* Branching Binary Influence (DBI)
'''

# used in get_NL, get_AI, get_NL to normalize the leadership values
def normalize_leadership(
        leadership : NDArray[Any], 
        norm_method : Literal[
            'max_leadership', 'crowd_size','max_min_leadership'
            ] = 'max_leadership'
        ) -> NDArray[Any]:
    '''
    Normalize the leadership values in a given network.

    Parameters
    -----
    leadership: ndarray of shape (N,)
        Leadership values.
        Nodes are in the order of the pedestrian ID order (0 to N-1).
    norm_method: str (optional)
        * if 'max_leadership' (default): values are normalized based on 
        the maximum leadership value in the given network
        * if 'crowd_size': normalized based on crowd size (N-1)
        * if 'max_min_leadership': positive values are normalized by 
        the maximum value, while negative values are normalized by the 
        absolute value of the minimum value

    Returns
    -----
    norm_leadership: ndarray of shape (N,)
        Normalized values
    '''
    if (norm_method=="max_leadership"):
        if max(leadership) == 0:
            norm_leadership = np.full_like(leadership, np.nan)
        else:
            norm_leadership = leadership / max(leadership)
    elif (norm_method=="crowd_size"):
        N = leadership.shape[0]
        norm_leadership = leadership / (N-1)
    elif (norm_method=="max_min_leadership"):
        norm_leadership = copy.copy(leadership).astype(np.float64)
        norm_leadership[leadership > 0] /= max(leadership)
        norm_leadership[leadership < 0] /= abs(min(leadership))

    return norm_leadership


def get_DI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal[
               'max_leadership', 'crowd_size','max_min_leadership'
           ] = 'max_leadership'
           ) -> NDArray[Any]: 
    '''
    Compute Direct Influence for all N pedestrians in a given network
    (DI; weighted outdegree - weighted indegree).

    Parameters
    -----
    weights: ndarray of shape (N, N)
        Network weights
    normalize : bool (optional)
        * if True (default): normalize values
        * if False: values are not normalized
    order : str (optional)
        The order of weights.
        * if 'ij' (default): the 1st dimension of weights (rows)  
        represents pedestrian i (leaders), and the 2nd dimension 
        (columns) represents  pedestrian j (followers).
        * if 'ji': the 1st dimension of weights (rows) represents 
        pedestrian j (followers), and the 2nd dimension (columns) 
        represents pedestrian i (leaders).
    norm_method: str (optional)
        * if 'max_leadership' (default): values are normalized based 
        on the maximum leadership value in the given network.
        * if 'crowd_size': normalized based on crowd size (N-1)
        * if 'max_min_leadership': positive values are normalized by  
        the maximum value, while negative values are normalized by the 
        absolute value of the minimum value

    Returns
    -----
    di: ndarray of DI values with shape (N,)
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)
    
    G = create_network(weights, order=order)     # create a directed graph
    weighted_outdegs = np.array(G.out_degree(weight='weight'))[:,1]     # (N,)
    weighted_indegs = np.array(G.in_degree(weight='weight'))[:,1]       # (N,)
    di = weighted_outdegs - weighted_indegs
    
    if normalize:
        di = normalize_leadership(di, norm_method)

    return di


def get_DBI(weights : NDArray[Any], 
            normalize : bool = True, 
            order : Literal["ij", 'ji'] = "ij", 
            norm_method : Literal[
                'max_leadership', 'crowd_size','max_min_leadership'
            ] = 'max_leadership'
            ) -> NDArray[Any]: 
    '''
    Compute Direct Binary Influence (DBI; outdegree - indegree) for all N 
    pedestrians in a given network.

    Parameters
    -----
    weights: ndarray of network weightswith shape (N, N)
    normalize : bool (optional)
        * if True (default): normalize values
        * if False: values are not normalized
    order : str (optional)
        Indicates the order of weights.
        * if 'ij' (default): the 1st dimension of weights (rows)  
        represents pedestrian i (leaders), and the 2nd dimension 
        (columns) represents  pedestrian j (followers).
        * if 'ji': the 1st dimension of weights (rows) represents 
        pedestrian j (followers), and the 2nd dimension (columns) 
        represents pedestrian i (leaders).
    norm_method: str (optional)
        * if 'max_leadership' (default): values are normalized based 
        on the maximum leadership value in the given network.
        * if 'crowd_size': normalized based on crowd size (N-1)
        * if 'max_min_leadership': positive values are normalized by  
        the maximum value, while negative values are normalized by the 
        absolute value of the minimum value
    
    Returns
    -----
    dbi: ndarray of DBI values with shape (N,)
    '''
    # replace NaN weights (missing) with 0 (no connections) so that it can
    # compute leadership properly. Otherwise in/out-degrees would be NaN.
    weights = np.nan_to_num(weights, nan=0)
    
    G = create_network(weights, order=order)     # create a directed graph
    outdegs = np.array(G.out_degree())[:,1]     # (N,)
    indegs = np.array(G.in_degree())[:,1]       # (N,)
    dbi = outdegs - indegs

    if normalize:
        dbi = normalize_leadership(dbi, norm_method)
    
    return dbi


def get_BI(weights : NDArray[Any], 
           normalize : bool = True, 
           order : Literal["ij", 'ji'] = "ij", 
           norm_method : Literal[
               'max_leadership', 'crowd_size','max_min_leadership'
           ] = 'max_leadership'
           ) -> NDArray[Any]: 
    '''
    Compute Branching Influence (BI; the sum of multiplied weights of each
    edge in each possible path between the pedestrians) for all N pedestrians 
    in a given network.

    Parameters
    -----
    weights: ndarray of network weightswith shape (N, N)
    normalize : bool (optional)
        * if True (default): normalize values
        * if False: values are not normalized
    order : str (optional)
        Indicates the order of weights.
        * if 'ij' (default): the 1st dimension of weights (rows)  
        represents pedestrian i (leaders), and the 2nd dimension 
        (columns) represents  pedestrian j (followers).
        * if 'ji': the 1st dimension of weights (rows) represents 
        pedestrian j (followers), and the 2nd dimension (columns) 
        represents pedestrian i (leaders).
    norm_method: str (optional)
        * if 'max_leadership' (default): values are normalized based 
        on the maximum leadership value in the given network.
        * if 'crowd_size': normalized based on crowd size (N-1)
        * if 'max_min_leadership': positive values are normalized by  
        the maximum value, while negative values are normalized by the 
        absolute value of the minimum value

    Returns
    -----
    bi: ndarray of BI values with shape (N,)
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
                w[i,j] = np.sum([ 
                    np.prod([weights[p[l+1], p[l]] for l in range(len(p)-1)])
                    for p in nx.all_simple_paths(G, source=i, target=j)
                ])
    bi = np.sum(w, axis=1)

    if normalize:
        bi = normalize_leadership(bi, norm_method)

    return bi


def get_DBI(weights : NDArray[Any], 
            normalize : bool = True, 
            order : Literal["ij", 'ji'] = "ij", 
            norm_method : Literal[
                'max_leadership', 'crowd_size','max_min_leadership'
            ] = 'max_leadership'
            ) -> NDArray[Any]: 
    '''
    Compute Branching Outdegree (DBI; the number of all possible paths 
    starting from a given pedestrian) for a given network.

    Parameters
    -----
    weights: ndarray of network weightswith shape (N, N)
    normalize : bool (optional)
        * if True (default): normalize values
        * if False: values are not normalized
    order : str (optional)
        Indicates the order of weights.
        * if 'ij' (default): the 1st dimension of weights (rows)  
        represents pedestrian i (leaders), and the 2nd dimension 
        (columns) represents  pedestrian j (followers).
        * if 'ji': the 1st dimension of weights (rows) represents 
        pedestrian j (followers), and the 2nd dimension (columns) 
        represents pedestrian i (leaders).
    norm_method: str (optional)
        * if 'max_leadership' (default): values are normalized based 
        on the maximum leadership value in the given network.
        * if 'crowd_size': normalized based on crowd size (N-1)
        * if 'max_min_leadership': positive values are normalized by  
        the maximum value, while negative values are normalized by the 
        absolute value of the minimum value

    Returns
    -----
    bbi: ndarray of BBI values with shape (N,)
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
        raise ValueError('Incorrect value for order while getting NL')

    # --- get paths ---
    bbi = np.zeros((N,))
    for i in range(N):
        for j in range(N):
            if (i!=j and nx.has_path(G, source=i, target=j)):
                bbi[i] += sum(
                    1 for path in nx.all_simple_paths(G, source=i, target=j))

    if normalize:
        bbi = normalize_leadership(bbi, norm_method)

    return bbi


def get_rank(leadership_value : NDArray[Any]) -> NDArray[Any]:
    '''
    Compute ranks for a specified leadership measure for each of N 
    pedestrians in a given network.

    Parameters
    -----
    leadership_value: ndarray of leadership values with shape (N,)

    Returns
    -----
    leadership_rank: ndarray of ranks with shape (N,)
    '''
    N = np.shape(leadership_value)[0]

    # replace NaNs with -np.inf so that they get the lowest ranking
    leadership_value = np.where(np.isnan(leadership_value), 
                                -np.inf, leadership_value)

    # sort the indices in descending order (higher value = more highly ranked)
    rank_ind = np.argsort(leadership_value)[::-1][:N]
    leadership_rank = np.argsort(rank_ind) + 1
    
    return leadership_rank
