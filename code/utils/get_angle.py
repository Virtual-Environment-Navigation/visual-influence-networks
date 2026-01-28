import numpy as np
from numpy.typing import NDArray
from typing import Any

'''
Functions related to angle calculations.

* heading2angle(heading) : North=0º (CW) to East=0º (CCW)
* angle2heading(angle) : East=0º (CCW) to North=0º (CW)
    - opposite from heading2angle
* angle2angle(angle) : change from [0, 360] to [-180, 180]
* angle2slope(angles) : angles of lines from x-axis (CCW) to slopes
* slope2angle(m) : slopes of lines to return angles from x-axis (CCW)
* get_angle_bw_slopes(m1, m2) : the angle between two lines from slopes
'''

def heading2angle(heading: NDArray[Any]) -> NDArray[Any]:
    '''
    Change angle system from North=0º (CW) to East=0º (CCW).

    Parameters
    ----------
    heading: ndarray of any shape
        Heading angles (degrees), where 0º is at North direction, and 
        angles increase clockwise, ranging between [0,360].

    Returns
    -------
    angle: ndarray of the same shape as heading
        Angles (degrees), where 0º is at East direction, and angles 
        increase counter-clockwise, ranging between [0,360].
    '''
    # rotate
    angle = np.where((-90<=heading), heading-90, heading-90+360)
    # flip (CW to CCW)
    angle = np.where(((angle==0) | (angle==180)), angle, -angle)
    return angle


def angle2heading(angle: NDArray[Any]) -> NDArray[Any]:
    '''
    Change angle system to same as heading. 
    This is the opposite function of heading2angle().

    Parameters
    ----------
    angle: ndarray of the same shape as heading
        Angles (degrees), where 0º is at East direction, and angles 
        increase counter-clockwise, ranging between [0,360].

    Returns
    -------
    new_angle: ndarray of any shape
        Heading angles (degrees), where 0º is at North direction, and 
        angles increase clockwise, ranging between [0,360].
    '''
    # flip (counterclockwise to clockwise)
    new_angle = np.where(((angle==0) | (angle==180)), angle, -angle)
    # rotate; [N, N, 2]
    new_angle = np.where((new_angle<=90), new_angle+90, new_angle+90-360)
    return new_angle


def angle2angle(angle: NDArray[Any]) -> NDArray[Any]:
    '''
    Modify angles ranging between [0, 360] (e.g., 220) to desired angles
    ranging between [-180, 180] (e.g., -40).
    
    Parameters
    ----------
    angle: ndarray of any shape
        Angles (degrees), where 0º is at East direction, and angles 
        increase counter-clockwise, ranging between [0,360].
    
    Returns
    -------
    new_angle: ndarray of any shape
        Modified angles (degrees), where 0º is at East direction, and 
        angles increase counter-clockwise, ranging between [-180,180].
    '''
    new_angle = np.where((angle>180), angle-360, angle)
    new_angle = np.where((new_angle<-180), angle+360, angle)
    return new_angle


def angle2slope(angle: NDArray[Any]) -> NDArray[Any]:
    '''
    Take angles (East = 0º, CCW) and return slopes.

    Parameters
    ----------
    angle: ndarray of the same shape as heading
        Angles (degrees), where 0º is at East direction, and angles 
        increase counter-clockwise, ranging between [0,360].
    
    Returns
    -------
    m:  ndarray of any shape
        Slopes of corresponding lines.
    '''
    m = np.tan(np.radians(angle))   # slopes for heading
    return m


def slope2angle(m: NDArray[Any]) -> NDArray[Any]:
    '''
    Take slopes of lines and return angles (East = 0º, CCW).

    Parameters
    ----------
    m:  ndarray of any shape
        Slopes of corresponding lines.

    Returns
    -------
    angle: ndarray of the same shape as m
        Angles (degrees) corresponding to the slopes, where 0º is at 
        East direction, and angles increase counter-clockwise, 
        ranging between [-90, 90].
    '''
    angle = np.rad2deg(np.arctan(m))
    return angle


def get_angle_bw_slopes(m1: NDArray[Any],
                        m2: NDArray[Any]) -> NDArray[Any]:
    '''
    Function to find the angle between two lines from slopes 
    (element-wise comparison).
    See https://www.geeksforgeeks.org/angle-between-a-pair-of-lines/).

    Parameters
    ----------
    m1, m2: ndarray of any shape
        Slopes of the two lines.

    Returns
    -------
    angle: ndarray of the same shape as m1 & m2
        Angles (degrees) between the two lines, ranging between [0,90].
    '''
    if m1.shape != m2.shape:
        raise ValueError(
            'The inputs should be the same shape: ', m1.shape, m2.shape
        )

    # tan value of the angle
    tan = np.absolute((m2 - m1) / (1 + m1 * m2))
    # tan inverse of the angle & convert radian to degree
    angle = np.rad2deg(np.arctan(tan))  
    
    return angle
