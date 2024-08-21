import numpy as np
from numpy.typing import NDArray
from typing import Union
from typing import Tuple        # find_tangent_lines()
from scipy.spatial import distance_matrix   # get_distance()
import warnings

from utils.get_angle import *

'''
Taken from occlusion_local_func.py in Research2021.
These are the local functions used in the later functions, which are from
occlusion_func.py.

* init_xy(x, y)
    - turn two arrays of x & y positions into one array
* get_present_agents(x)
    - get a list of pedestrians not missing
* find_tangent_lines(center, semi_axes, rotation, reference_point)
    - find the Ellipse's two tangents that go through a reference point
    - Used in get_occlusion()
* get_quadrant(x, y, tanX, tanY)
    - find quadrant of points based on another point as a center
    - Determine quadrant of j (viewed) with viewer i as origin
    - Used in get_occlusion()
* get_angle_quadrant(m, quadrant)
    - Find angles of lines based on quadrant of a point on the line.
    - Used in get_occlusion()
* is_point_within_view(heading, tan_angle, view_angle)
    - Determine whether (tangent) points are within field of view, based on the xy positions.
    - Used in get_view_angle()
* get_angles_in_view(heading, m, m_view_reshaped, tan_angle, original_angle, view_angle=180)
    - Compute view angle of each pedestrian considering field of view
    - Used in get_view_angle()
* get_occluded_angles(heading, angle_original, angle_FOV, 
                      m_FOV, tan_angle, dist)
    - get view angles after considering field of view and occlusion
    - Used in get_view_angle()

'''

def init_xy(x : NDArray[np.number], 
            y : NDArray[np.number]) -> NDArray[np.number]:
    '''
    Turn two arrays of x & y positions into one array.
    
    Parameters
    -----
    x, y : numpy arrays of float
        Shape (num_datapoints,).
        x & y positions of a given pedestrian.

    Returns
    -----
    xy : numpy array of float
        Shape (num_datapoints, 2).
        the combined xy positions.
    '''
    x_reshaped = x.reshape(len(x), 1)
    y_reshaped = y.reshape(len(x), 1)
    xy = np.concatenate((x_reshaped, y_reshaped), axis=1)

    return xy


def get_present_agents(x : NDArray[np.number]) -> NDArray[np.integer]:
    '''
    In a given set of x positions for N pedestrians, get a list of pedestrians
    who are present in the data (not missing).

    Parameters
    -----
    x : numpy array of float
        Shape (N,).
        x positions of N pedestrians.

    Returns
    -----
    ind : numpy array of int
        Shape (num_present_pedestrians,). 
        A list of indices (= pedestrian_ID - 1) of pedestrians who are not missing in this dataset
    '''
    ind = np.argwhere(~np.isnan(x)).flatten()    # remove agents without data
    return ind


# Used in get_occlusion()
def find_tangent_lines(center: Tuple[float, float],
                       semi_axes: Tuple[float, float],
                       rotation: float,
                       reference_point: Tuple[float, float]
                       ) -> tuple:
    '''
    Find the Ellipse's two tangents that go through a reference point.
    (added tangent points to: https://stackoverflow.com/a/70849492/8229013)
    Used in get_occlusion().

    Parameters
    -----
    center : a tuple of float
        The coordinates of the center of the ellipse.
    semi_axes : a tuple of float
        The semi-major and semi-minor axes of the ellipse.
    rotation : float
        The counter-clockwise rotation of the ellipse in radians.
    reference_point : a tuple of float
        The coordinates of the reference point.

    Returns
    -----
    (m1, h1, x1, y1) : tuple
        Slope(m), intercept(h), and tangent points (x,y) of the first tangent.
    (m2, h2, x2, y2) : tuple
        Slope(m), intercept(h), and tangent points (x,y) of the second tangent.
    '''
    x0, y0 = center
    a, b = semi_axes
    s, c = np.sin(rotation), np.cos(rotation)
    p0, q0 = reference_point

    A = y0 - q0
    D = x0 - p0
    B = a*s
    E = a*c
    C = b*c
    F = -b*s
    denominator = np.sqrt((C*D - A*F)**2 + (A*E - B*D)**2)

    beta = np.arctan2( (C*D-A*F)/denominator,  (A*E-B*D)/denominator)
    theta = [ -beta+np.arcsin((B*F-C*E)/denominator), 
              -beta-np.arcsin((B*F-C*E)/denominator)+np.pi ]
    p_ellipse = [( x0+E*np.cos(t)+F*np.sin(t), y0+B*np.cos(t)+C*np.sin(t) ) 
                 for t in theta ]
    slope = [ (e[1]-reference_point[1])/(e[0]-reference_point[0]) 
             for e in p_ellipse ]
    intercept = [ e[1]-s*e[0] for e, s in zip(p_ellipse, slope) ]
    tangent_point_x = [ e[0] for e in p_ellipse ]
    tangent_point_y = [ e[1] for e in p_ellipse ]

    return (
        (slope[0], intercept[0], tangent_point_x[0], tangent_point_y[0]),
        (slope[1], intercept[1], tangent_point_x[1], tangent_point_y[1]),
    )


# Used in get_occlusion()
def get_quadrant(x : NDArray[np.number], 
                 y : NDArray[np.number], 
                 tanX : NDArray[np.number], 
                 tanY : NDArray[np.number]) -> NDArray[np.number]:
    '''
    Find quadrant of points based on another point as a center.
    Determine quadrant no. of j (viewed) with viewer i as origin (i x j).
    Used to compute angles.

    Parameters
    -----
    x, y : numpy array of float
        Shape (N,).
        x & y positions of all pedestrians.
    tanX, tanY : numpy array of float
        Shape (N, N, 2).
        x & y tangent points.
    
    Returns
    -----
    quadrant : numpy array of int
        Shape (N, N, 2). 
        Represents quadrant 1-4; 0 for itself; nan if missing
    '''
    # reshape x & y to make them into the same shape as tanX & tanY
    N = np.shape(x)[0]
    reshapedX = np.repeat(x, N).reshape((N, N))
    reshapedY = np.repeat(y, N).reshape((N, N))
    reshapedX = np.repeat(reshapedX[:, :, np.newaxis], 2, axis=2)
    reshapedY = np.repeat(reshapedY[:, :, np.newaxis], 2, axis=2)
    # get dx & dy (=distance between center and tangent points)
    dx = tanX - reshapedX   # (N x N x 2)
    dy = tanY - reshapedY
    # location of the tangent point (tanX & tanY) wrt the center (i; x & y)
    mask_dx = np.where(dx>=0, True, False)   # True = positive; (N x N x 2)
    mask_dy = np.where(dy>=0, True, False)
    # compute quadrant
    quadrant = np.ones([N, N, 2])           # Quadrant I (NE)
    quadrant[~mask_dx & mask_dy] = 2        # II (NW)
    quadrant[~mask_dx & ~mask_dy] = 3       # III (SW)
    quadrant[mask_dx & ~mask_dy] = 4        # IV (SE)
    quadrant[np.diag_indices(N)] = 0        # set diagonal as 0
    quadrant[np.isnan(reshapedX)] = np.nan  # set nan
    
    return quadrant


# Used in get_occlusion()
def get_angle_quadrant(m : NDArray[np.number], 
                       quadrant : NDArray[np.integer]) -> NDArray[np.number]:
    '''
    Find angles of lines based on quadrant of a point on the line.
    Change [-90, 90] to [-180, 180].

    Parameters
    -----
    m : numpy array of floats
        Shape (N, N, 2). Represents slopes.
    quadrant : numpy array of integers 
        Shape (N, N, 2). Used as mask.

    Returns
    -----
    angle : numpy array of floats 
        Shape (N, N, 2) = [i,j,?], with i as a center, the angle of the 
        line/tangent point from x axis.
    '''
    angle = np.rad2deg(np.arctan(m))     # degree of all the lines from x axis [-90, 90]
    angle[quadrant==2] = angle[quadrant==2]+180      # II (NW)
    angle[quadrant==3] = angle[quadrant==3]-180      # II (SW)
    angle[np.isnan(quadrant)] = np.nan
    return angle

# Used in get_view_angle()
def is_point_within_view(heading : NDArray[np.number], 
                         tan_angle : NDArray[np.number], 
                         view_angle : Union[int, float] = 180
                         ) -> NDArray[np.bool_]:
    '''
    Determine whether (tangent) points are within field of view, based on the xy positions.

    Parameters
    -----
    heading : numpy array of floats 
        Shape (N,). 
        *CAUTION* values increase clockwise (in degrees).
    tan_angle : numpy array of floats 
        Shape (N, N, 2). 
        Represents angles between tangent points and x-axis with viewer i as a 
        center (i x j x 2 lines).
        *CAUTION* angles increase counterclockwise (in degrees).

    [Optional]
    view_angle (default = 180) : float
        Assumed field of view in degree.

    Returns
    -----
    isWithinView : numpy array of bool
        Shape (N, N, 2) = (viewer i, viewed j, 2 lines).
        * if True -> a point is within field of view
        * if False -> a point is NOT in view. 
    '''
    # init
    N = len(heading)
    # change to our clockwise direction system ([North,W,S,E] = [0,-90,180,90])
    tan_angle = angle2heading(tan_angle)
    # get beta (angle between heading and tangent points)
    beta = np.array([np.absolute(tan_angle[i]-heading[i]) 
                     for i in range(N)])   # get difference
    beta = np.where((beta>180), np.absolute(beta-360), beta)
    # based on beta, determine whether within the field of view
    isWithinView = np.where((beta<view_angle/2), True, False)   # (N,N,2)
    # ignore the angle & beta from i to i
    di = np.diag_indices(N) # create a set of indices to access the diagonal (2, N)
    isWithinView[di[0],di[1],:] = False

    return isWithinView

# used in get_view_angle()
def get_angles_in_view(heading : NDArray[np.number], 
                       m : NDArray[np.number], 
                       m_view_reshaped : NDArray[np.number], 
                       tan_angle : NDArray[np.number], 
                       original_angle : NDArray[np.number],
                       view_angle : float = 180, 
                       ) -> tuple[NDArray[np.number], NDArray[np.number]]:
    '''
    Compute view angle of each pedestrian considering field of view

    Parameters
    -----
    heading : numpy array of float
        Shape (N,). 
        Heading of each pedestrian.
    m : numpy array of float
        Shape (N, N, 2). 
        Represents slope for each tangent line.
    m_view_reshaped : numpy array of float
        Shape (N, N).
        Slopes of field of view (N,), but reshaped into (N, N). 
    tan_angle : numpy array of floats 
        Shape (N, N, 2). 
        Represents angles between tangent points and x-axis with viewer i as a 
        center (i x j x 2 lines).
        *CAUTION* angles increase counterclockwise (in degrees).
    original_angle : numpy array of floats
        Shape (N, N). 
        View angle without considering occlusion or field of view.

    [Optional]
    view_angle (default = 180) : int or float
        field of view in degree 

    Returns
    -----
    new_angle : numpy array of float
        Shape (N, N). 
        View angle after considering occlusion & field of view .
    new_m : numpy array of float
        Shape (N, N, 2).
        Slope for lines creating visual angle.
        Tangent line is replaced with the line for field of vision when visual 
        angle is changed by it.
    new_tan_angle : numpy array of float
        Shape (N, N, 2). 
        Angles between tangent point and x axis with i (viewer) as a center. 
        Similarly to new_m, angle is replaced when field of vision is affecting 
        the visual angle.
    '''
    # init
    N = len(heading)
    new_angle = np.zeros((N,N))  # store final angle
    new_m = np.copy(m)
    # new_tan_angle = np.copy(tan_angle)
    isWithinView = is_point_within_view(heading, tan_angle, view_angle)     # True if a tangent point is within view (N, N, 2)

    # check if agents are within field of view based on tangent points
    isBothWithinView = np.logical_and(isWithinView[:,:,0], isWithinView[:,:,1]) # True if both tangent points are within view (FULLY) [N,N]
    isOneWithinView = np.logical_or(isWithinView[:,:,0], isWithinView[:,:,1])   # True if at least one tangent point is within view  [N,N]
    isOnlyOneWithinView = np.logical_and(~isBothWithinView, isOneWithinView)    # True if only one tangent point is within view (PARTIALLY) [N,N]
    isNotWithinView = np.logical_not(isOneWithinView)   # True if only one tangent point is within view (NONE) [N,N]

    # when both tangent points are in field of view, apply original angle
    new_angle[isBothWithinView] = original_angle[isBothWithinView]

    # apply original angle (=nan) for missing data
    new_angle[np.isnan(original_angle)] = np.nan

    # when only one tangent point is in field of view, compute & apply new angle
    tan_index = np.where((isWithinView[:,:,0]==True), 0, 1)    # True if 1st tangent point is in view; otherwise False [N,N]
     # if only 1st tangent point is in view
    new_angle[isOnlyOneWithinView & (tan_index==0)] = get_angle_bw_slopes(m_view_reshaped, m[:,:,0])[isOnlyOneWithinView & (tan_index==0)]  # [N,N]
    new_m[:,:,1][isOnlyOneWithinView & (tan_index==0)] = m_view_reshaped[isOnlyOneWithinView & (tan_index==0)]    # replace slope of tangent line with slope of field of view

     # if only 2nd tangent point is in view
    new_angle[isOnlyOneWithinView & (tan_index==1)] = get_angle_bw_slopes(m_view_reshaped, m[:,:,1])[isOnlyOneWithinView & (tan_index==1)]  # [N,N]
    new_m[:,:,0][isOnlyOneWithinView & (tan_index==1)] = m_view_reshaped[isOnlyOneWithinView & (tan_index==1)]    # replace slope of tangent line with slope of field of view

    return new_angle, new_m


# used in get_view_angle()
def get_occluded_angles(heading : NDArray[np.number], 
                        angle_FOV : NDArray[np.number], 
                        m_FOV : NDArray[np.number], 
                        tan_angle : NDArray[np.number], 
                        dist : NDArray[np.number]
                        ) -> tuple[NDArray[np.number]]:
    '''
    Get view angles after considering field of view and occlusion.
    Used in get_view_angle().

    Parameters
    -----
    heading : numpy arary of float
        Shape (N,).
    angle_FOV: numpy arary of float
        Shape (N, N). 
        View angles where field of view is considered, but not occlusion.
    m_FOV : numpy arary of float
        Shape (N, N, 2).
        Slopes for tangenet lines where field of view is considered.
    tan_angle : numpy array of float
        Shape (N, N, 2). 
        Angles between tangent point and x axis with i (viewer) as a center. 
        - original; without consideration of field of view or occlusion
    dist : numpy arary of float
        Shape (N, N).
        Indicates distance.

    Returns
    -----
    angle_visual : numpy arary of float
        Shape (N, N).
        View angles where field of view and occlusion are considered.
    m_visual : numpy arary of float
        Shape (N, N, 2).
        Slopes for visual lines making up visual angle where occlusion is 
        considered.
    '''
    # init
    N = len(heading)
    angle_visual = np.copy(angle_FOV)     # store final angle
    m_visual = np.copy(m_FOV)

    for i in range(N):  # viewer
        visible_neighbors = np.argwhere(np.logical_and(angle_FOV[i,:]!=0, 
                                                       ~np.isnan(angle_FOV[i,:]))
                                                       ).flatten()  # not 0 or nan
        if visible_neighbors.size == 0 or visible_neighbors.size == 1:
            continue
        visible_neighbors_dist = dist[i, visible_neighbors]
        sorted_visible_neighbors = visible_neighbors[np.argsort(visible_neighbors_dist)]    # from closest to furthest

        # change angles between [-90,90], with heading as 0
        temp_tan_angle = angle2angle(tan_angle[i]-heading2angle(heading[i])) 
        temp_tan_angle[i,:] = 0
        tan_angle_min = np.min(temp_tan_angle, axis=1)   # (N,); to the right from the viewer
        tan_angle_max = np.max(temp_tan_angle, axis=1)   # (N,); to the left from the viewer

        # is j occluded by k?
        for j_idx in range(len(sorted_visible_neighbors)):
            j = sorted_visible_neighbors[j_idx]
            j_min = tan_angle_min[j]
            j_max = tan_angle_max[j]

            for k_idx in range(j_idx):  # all pedestrians closer than j
                if (angle_visual[i,j] == 0):
                    break
                k = sorted_visible_neighbors[k_idx]
                k_min = tan_angle_min[k]
                k_max = tan_angle_max[k]

                # j is fully occluded
                if ((k_min <= j_min) and (j_max <= k_max)):   
                    angle_visual[i,j] = 0
                    break
                # j's left is occluded
                elif ((k_min < j_max) and (j_max <= k_max)):    
                    temp = get_angle_bw_slopes(m_visual[i,k,0], m_visual[i,j,1])
                    if (temp<angle_visual[i,j]):
                        m_visual[i,j,1] = m_visual[i,k,0]
                        angle_visual[i,j] = temp
                # j's right is occluded
                elif (k_min<=j_min and j_min<k_max):    
                    temp = get_angle_bw_slopes(m_visual[i,k,1], m_visual[i,j,1])
                    if (temp<angle_visual[i,j]):
                        m_visual[i,j,0] = m_visual[i,k,1]
                        angle_visual[i,j] = temp
                # j's middle is occluded
                elif (j_min<k_min and k_max<j_max):    
                    angle_visual[i,j] = 0
                    break

    return angle_visual, m_visual


# =============================================================================
'''
Taken from occlusion_func.py in Research2021.
These are the local functions used in the later functions, which are from
occlusion_func.py.

* get_distance(x, y)
    - Compute the distance matrix.
* get_FOV(x, y, heading, view_angle=180)
    - Determine who is within field of view, based on their xy positions.
* get_view_tangent(x, y, heading, agent_width, agent_depth)
    - Compute components of tangent lines.
* get_view_angle(heading, m, tan_angle, view_angle, dist)
    - Compute view angle of each pedestrian.

'''

def get_distance(x : NDArray[np.number], 
                 y : NDArray[np.number]) -> NDArray[np.number]:
    '''
    Compute the distance matrix.

    Parameters
    -----
    x : numpy array of float
        Shape (N,).
        x positions for all pedestrians.
    y : numpy array of float
        Shape (N,).
        y positions for all pedestrians.

    Returns:
    -----
    dist_all : numpy array of float
        Shape (N, N). 
        Matrix of all pair-wise distances.
    '''
    # init
    xy = init_xy(x, y)
    ## get distance (return 0 for itself; nan when one is nan)
    dist_all = distance_matrix(xy, xy)  # (N, N)
    return dist_all


def get_FOV(x : NDArray[np.number],
            y : NDArray[np.number],
            heading : NDArray[np.number],
            view_angle : Union[int, float] = 180) -> NDArray[np.number]:
    '''
    Determine who is within field of view, based on their xy positions.

    Parameters
    -----
    x : numpy array of float
        Shape (N,).
        x positions for all pedestrians.
    y : numpy array of float
        Shape (N,).
        y positions for all pedestrians.
    heading : numpy array of float
        Shape (N,).
        heading directions for all pedestrians.

    [Optional]
    view_angle (default = 180) : int or float
        assumed field of view.

    Returns:
    -----
    isWithinView : numpy array of bool
        Shape (N, N) = (viewer, viewed).
        * if True -> within field of view
        * if False -> not within field of view
    '''
    N = len(x)
    # get counterclockwise angle between x-axis and the line between focal and other agents ([East,N,W,S] = [0,90,180,-90])
    angle = np.array([np.arctan2(y-y[i], x-x[i])*180/np.pi for i in range(N)])  # for each focal agent i; in degrees [N x N]
    # change to our clockwise direction system ([North,W,S,E] = [0,-90,180,90])
    angle = np.where(((angle==0) | (angle==180)), angle, -angle) # flip (counterclockwise to clockwise)
    angle = np.where((angle<=90), angle+90, angle+90-360)   # rotate
    # get beta (angle from heading to other agents)
    beta = np.array([np.absolute(angle[i]-heading[i]) for i in range(N)])   # get difference
    beta = np.where((beta>180), np.absolute(beta-360), beta)
    # based on beta, determine whether within the field of view
    isWithinView = np.where((beta<view_angle/2), True, False)   # (N,N)
    # ignore the angle & beta from i to i
    di = np.diag_indices(N) # create a set of indices to access the diagonal
    isWithinView[di] = False

    return isWithinView


def get_view_tangent(x : NDArray[np.number], 
                     y : NDArray[np.number], 
                     heading : NDArray[np.number], 
                     agent_width : float = 0.45, 
                     agent_depth : float = 0.244) -> tuple:
    '''
    Compute components of tangent lines.

    Parameters
    -----
    x, y, heading : numpy array of float 
        Shape (N,).
    
    [Optional]
    agent_width (default = 0.45) : float
        ellipse width (major axis)
    agent_dept (default = 0.244) : float
        ellipse depth (minor axis)

    Returns
    -----
    m, h : numpy arrays of float
        Shape (N, N, 2) = (viewer i, viewed j, 2 tangent lines).
        Slopes (m) & intercepts (h) of the tangent lines.
    tanX, tanY : numpy array of float
        Shape (N, N, 2) = (viewer i, viewed j, 2 tangent lines).
        x & y positions of the tangent points.
    angle : numpy array of float
        Shape (N, N, 2) = (viewer i, viewed j, 2 tangent lines).
        Angles between tangent point and x axis with i (viewer) as a center.
    '''
    ## compute occlusion
    # init
    N = len(x)
    xy = init_xy(x, y)
    present_agents = get_present_agents(x)

    m = np.zeros([N, N, 2])     # slopes: i (viewer) x j (being viewed) x 2 lines (m1, m2)
    h = np.zeros([N, N, 2])     # intercept: i x j x 2 lines (h1, h2)
    tanX = np.zeros([N, N, 2])  # x for tangent point: i x j x 2 (x1, x2)
    tanY = np.zeros([N, N, 2])  # y for tangent point: i x j x 2 (y1, y2)

    # determine viewing angles
    for j in range(N):    # focal agent (viewed)
        if j not in present_agents:     # missing subjects
            m[:,j,:], h[:,j,:], tanX[:,j,:], tanY[:,j,:] = np.nan, np.nan, np.nan, np.nan
            continue
        CENTER = x[j], y[j]
        SEMI_AXES = agent_width/2, agent_depth/2
        ROTATION = np.radians(-heading[j])

        for i in range(N):    # viewer
            if i not in present_agents:     # missing subjects
                m[i,j,:], h[i,j,:], tanX[i,j,:], tanY[i,j,:] = np.nan, np.nan, np.nan, np.nan
            elif i!=j:
                (m[i,j,0], h[i,j,0], tanX[i,j,0], tanY[i,j,0]), (m[i,j,1], h[i,j,1], tanX[i,j,1], tanY[i,j,1])= find_tangent_lines(
                    center=CENTER,
                    semi_axes=SEMI_AXES,
                    rotation=ROTATION,
                    reference_point=xy[i] )

    # get quadrant of j with i as center
    quadrant = get_quadrant(x, y, tanX, tanY)   # (N,N,2)
    # get angle of 2 lines with i as center
    angle = get_angle_quadrant(m, quadrant)   # (N,N,2)

    # reorder the 2 lines based on angles (min, max)
    ind = np.argsort(angle, axis=2)
    m = np.take_along_axis(m, ind, axis=2)
    h = np.take_along_axis(h, ind, axis=2)
    tanX = np.take_along_axis(tanX, ind, axis=2)
    tanY = np.take_along_axis(tanY, ind, axis=2)
    angle = np.sort(angle, axis=2)

    return m, h, tanX, tanY, angle


def get_view_angle(heading : NDArray[np.number], 
                   m : NDArray[np.number], 
                   tan_angle : NDArray[np.number], 
                   dist : NDArray[np.number],
                   view_angle : Union[int, float] = 180) -> NDArray[np.number]:
    '''
    Compute view angle of each pedestrian.
    1. without considering occlusion / field of view
    2. after considering occlusion & field of view

    Parameters
    -----
    heading : numpy array of float 
        Shape (N,).
    m : numpy array of float
        Shape (N, N, 2).
        Slopes for each tangent line.
    tan_angle : numpy array of float
        Shape (N, N, 2).
        Angles between tangent point and x axis with i (viewer) as a center.
        Calculated in get_view_tangent().
    dist : numpy array of float 
        Shape (N, N).
        Calculated in get_distance().
        
    [Optional]
    view_angle (default = 180) : int or float 
        field of view in degree.

    Returns
    -----
    original_angle : numpy array of float
        Shape (N, N).
        View angle without considering occlusion or field of view.
    new_angle : numpy array of float
        Shape (N, N).
        View angle after considering field of view & occlusion.
    '''
    ## initialize slopes for heading & field of view
    N = len(heading)
    m_heading = angle2slope(heading2angle(heading))     # slopes of heading (N,)
    m_view = -1/m_heading       # slopes of field of view (N,)
    m_view_reshaped = np.repeat(m_view, N).reshape((N, N))  # (N, N)

    ## get view angle without considering occlusion / field of view
    original_angle = np.absolute(tan_angle[:,:,1]-tan_angle[:,:,0])
    original_angle = np.where(original_angle>180, np.absolute(360-original_angle), original_angle)  # (N, N)

    ## consider field of view (similar to get_FOV(), but check for tangent points instead of center)
    new_angle0, new_m0 = get_angles_in_view(heading, m, m_view_reshaped, 
                                            tan_angle, original_angle, 
                                            view_angle=view_angle) # (N, N)

    ## consider occlusion
    # check for occlusion where angle is not 0 or nan in new_angle0
    new_angle, m_visual = get_occluded_angles(heading, new_angle0, new_m0, tan_angle, dist)

    return original_angle, new_angle


# =============================================================================
'''
compute_all() from occlusion.py in Research2021.

* get_visibility_t()
    - get visibilities for all pedestrians at a given time point
* get_visibility_all()
    - get visibilities for all pedestrians at all time points

'''

def get_visibility_t(x : NDArray[np.number],
                     y : NDArray[np.number], 
                     heading : NDArray[np.number], 
                     agent_width : float = 0.45, 
                     agent_depth : float = 0.244, 
                     view_angle : Union[int, float] = 180) -> NDArray[np.number]:
    '''
    A function which determines occlusions in the crowd at a time point t.
    Assume that the agents are lines.

    heading
       ↑
    ———◯———
    

    Parameters
    -----
    x, y, heading : numpy array of float
        Shape (N,).
        Respectively, x, y positiosn & heading (clockwise) of N pedestrians.

    [Optional]
    agent_width (default = 0.45) : float
        assumed shoulder width of agents (in meters)
    agent_depth (default = 0.244) : float
        assumed depth (from chest to back) of agents (in meters)
    view_angle (default = 180) : int or float
        assumed field of view (in degrees)

    Returns
    -----
    visibility_mat : numpy array of float
        Shape (N, N) = (viewer i, viewed j).
        Matrix of visibility for each pair, ranging [0,1].
        * 0 if outside of view or completely occluded or missing
        * 1 if fully visible
        * otherwise: partial visibility of the agent j to agent i
    '''
    ## ===== init =====
    N = len(x)
    x = np.array(x)
    y = np.array(y)
    visibility_mat = np.zeros([N, N])

    dist = get_distance(x, y)   # (N, N)
    m, _, _, _, tan_angle = get_view_tangent(x, y, heading, 
                                             agent_width=agent_width, 
                                             agent_depth=agent_depth) # [N,N,2]
    original_angle, actual_angle = get_view_angle(heading, m, tan_angle, dist,
                                                  view_angle=view_angle)   # [N,N]

    ## ===== suppress warnings =====
    # RuntimeWarning: invalid value encountered in true_divide
    warnings.filterwarnings("ignore")
    visibility_mat[~np.isnan(original_angle) & (original_angle!=0)] = (actual_angle/original_angle)[~np.isnan(original_angle) & (original_angle!=0)]

    return visibility_mat

def get_visibility_all(x : NDArray[np.number],
                       y : NDArray[np.number],
                       heading : NDArray[np.number],
                       agent_width=0.45, 
                       agent_depth=0.244, 
                       view_angle=180) -> NDArray[np.number]:
    '''
    compute visibility for each time point.

    Parameters
    -----
    x, y, heading : numpy array of float
        Shape (num_timepoints, N).
        Respectively, x, y positiosn & heading (clockwise).
        NOTE: x, y, & heading all need to have the same shape

    [Optional]
    agent_width (default = 0.45) : float
        assumed shoulder width of agents (in meters)
    agent_depth (default = 0.244) : float
        assumed depth (from chest to back) of agents (in meters)
    view_angle (default = 180) : int or float
        assumed field of view (in degrees)

    Returns
    -----
    visibility_mat : numpy array of float
        Shape (num_timepoints, N, N) = [t, viewer i, viewed j].
        Visibility for each pair, values ranging between [0,1].
    '''
    num_timepoints, N = np.shape(x)
    visibility_mat = np.empty([num_timepoints, N, N])

    for t in range(num_timepoints):
        visibility_mat[t,:,:] = get_visibility_t(x[t], y[t], heading[t],
                                                 agent_width=agent_width,
                                                 agent_depth=agent_depth,
                                                 view_angle=view_angle)
    
    if (np.any(visibility_mat)>1) or (np.any(visibility_mat)<0):
        raise ValueError('Something went wrong with the visibility calculation')

    return visibility_mat