import numpy as np
from numpy.typing import NDArray
from typing import Any, Union, Literal

'''
Functions to get the data needed for later analysis

* get_velocity()
* get_acceleration()
* get_speed()
* get_heading()
* get_mean_angle()
* transform_XY()
* get_relative_pos()
'''

def get_velocity(pos : NDArray[Any], 
                 SAMP_FREQ : Union[int, float], 
                 same_shape : bool = True,
                 insert_row: Literal['zeros', 'first_row'] = 'zeros') -> NDArray[Any]:
    '''
    Compute velocity from positions.

    Parameters
    -----
    pos : numpy array of positions
        The positions may be on a given axis, shaped (num_timepoints, N), 
        or on two axes, shaped (num_timepoints, N, 2).
    SAMP_FREQ : sample frequency (Hz)

    [Optional]
    same_shape (default = True) : bool
        It indicates if vel should be the same shape as pos.
        * if True -> a row of zeros is added at vel[0]
        * if False -> vel would be different from pos

    insert_row (default = 'zeros') : str
        It indicates what values should be inserted into the first row. 
        It is only used if same_shape is True.
        * if 'zeros' -> a row of zeros is inserted
        * if 'first_row' -> the same values are repeated

    Returns
    -----
    vel: numpy array of velocities
        By default, the same shape as pos, and the values of the first row are
        zeros. The shape may be different depending on same_shape & insert_row.
    '''
    dt = 1/SAMP_FREQ    # time steps (in s)
    # calculate velocities
    # shape: similar to pos but the 1st dimension is num_timepoints-1
    # (num_timepoints-1, N), or (num_timepoints-1, N, 2).
    vel = np.diff(pos, axis=0) / dt

    if (same_shape==True):
        if (vel.ndim == 1):     # if pos is 1D
            new_value = 0 if (insert_row == 'zeros') else vel[0]
            vel = np.insert(vel, 0, new_value)  # add value at the beginning
        else:   # if pos is 2D or more
            new_row = np.zeros_like(vel[0]) if (insert_row == 'zeros') else vel[0]
            vel = np.vstack((new_row, vel))
    
    return vel

def get_acceleration(vel : NDArray[Any], 
                     SAMP_FREQ : Union[int, float], 
                     same_shape : bool = True,
                     insert_row: Literal['zeros', 'first_row'] = 'zeros') -> NDArray[Any]:
    '''
    Compute acceleration from velocities.

    Parameters
    -----
    vel: numpy array of velocities
        The velocities may be on a given axis, shaped (num_timepoints, N), 
        or on two axes, shaped (num_timepoints, N, 2).
        See the description of pos & vel in get_velocity().
    SAMP_FREQ : sample frequency (Hz)

    [Optional]
    same_shape (default = True) : bool
        It indicates if vel should be the same shape as pos.
        * if True -> a row of zeros is added at vel[0]
        * if False -> vel would be different from pos

    insert_row (default = 'zeros') : str
        It indicates what values should be inserted into the first row. 
        It is only used if same_shape is True.
        * if 'zeros' -> a row of zeros is inserted
        * if 'first_row' -> the same values are repeated

    Returns:
    -----
    acc: numpy array of accelerations
        By default, the same shape as vel, and the values of the first row are
        zeros. The shape may be different depending on same_shape & insert_row.
    '''
    dt = 1/SAMP_FREQ    # time steps
    # calculate accelerations (ignore the first row with zeros)
    acc = np.diff(vel[1:], axis=0) / dt

    if (same_shape==True):
        if (vel.ndim == 1):     # if pos is 1D
            new_value = 0 if (insert_row == 'zeros') else acc[0]
            acc = np.insert(acc, 0, new_value)  # add value at the beginning
        else:   # if pos is 2D or more
            new_row = np.zeros_like(acc[0]) if (insert_row == 'zeros') else acc[0]
            acc = np.vstack((new_row, acc))

    return acc

def get_speed(x : NDArray[Any], 
              y : NDArray[Any], 
              SAMP_FREQ : Union[int, float]) -> NDArray[Any]:
    '''
    Compute speed from XY positions.

    Parameters
    -----
    x : numpy array of X positions
        Any shape.
    y : numpy array of X positions
        Any shape.
    SAMP_FREQ : sample frequency (Hz)

    Returns
    -----
    speed : numpy array of speed
        Same shape as x & y.
    '''
    vx = np.diff(x)
    vy = np.diff(y)

    speed = np.sqrt(vx**2 + vy**2)
    speed = np.concatenate([speed, [speed[-1]]])
    speed *= SAMP_FREQ   # Convert to proper units 

    return speed


def get_heading(x : NDArray[Any], 
                y : NDArray[Any]) -> NDArray[Any]:
    '''
    Produce speed and heading time series from time series of position (x,y). 
                  0
                  ^
                  |    
         -90  ————————> +90
                  |
                  |
                 ±180

    Parameters
    -----
    x : numpy array of X positions
        Any shape.
    y : numpy array of X positions
        Any shape.

    Returns
    -----
    speed : numpy array of speed
        Same shape as x & y. 0º is set as North direction, ranging between 
        [-180,180].
    '''
    vx = np.diff(x)
    vy = np.diff(y)

    # cart2pol() in Matlab
    heading = np.arctan2(vy, vx)
    heading = np.concatenate([heading, [heading[-1]]])
    # Convert to proper units
    heading = np.degrees(-heading + np.radians(90)) 
    # Rotate polar to cartesian coordinates so they equal straight head 
    # / upward direction as 0 degrees
    heading[heading > 180] -= 360

    return heading

def get_mean_angle(angles : NDArray[Any], 
                   angle_type : Literal['degree', 'radian'] = 'degree',
                   sens : float = 1e-12) -> float:
    '''
    Calculate the mean of a set of angles (in degrees) based on polar considerations.

    Parameters
    -----
    angles: numpy array of of angles

    [Optional]
    angle_type : str
        'degree' (default) or 'radian'
    sens: Sensitivity factor to determine oppositeness. Default is 1e-12.

    Returns:
    -----
    mean_angle: float
        Mean of the angles. The unit is the same as the input angles.
    '''
    if (angle_type=='degree'):
        angles = np.deg2rad(angles)
    
    angles = np.exp(1j * angles)
    mid = np.nanmean(angles)
    mean_angle = np.arctan2(np.imag(mid), np.real(mid))     # in radian

    if (angle_type=='degree'):
        mean_angle = np.rad2deg(mean_angle)

    return mean_angle


def transform_XY(x : NDArray[Any], 
                 y : NDArray[Any],
                 heading : NDArray[Any],
                 angle_type : Literal['degree', 'radian'] = 'degree') -> NDArray[Any]:
    '''
    Transform the coordinates based on the mean heading.

    Parameters
    -----
    x, y : numpy array of x & y positions
        Any shape.
    heading : numpy array of heading
        Any but the same shape as x & y. 
        The unit is specified by angle_type.

    [Optional]
    angle_type : str
        'degree' (default) or 'radian'

    Returns
    -----
    x_rotated, y_rotated : numpy array of the new x & y positions
    '''
    # theta : rotation angle (in degrees) = group mean angle
    theta = get_mean_angle(heading, angle_type=angle_type)
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    
    # format the data & deal with missing data
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    points = np.vstack((x[valid_mask], y[valid_mask]))

    # rotate
    points_rotated = np.dot(R, points)
    x_rotated = np.full_like(x, np.nan)
    y_rotated = np.full_like(y, np.nan)
    x_rotated[valid_mask] = points_rotated[0, :]
    y_rotated[valid_mask] = points_rotated[1, :]

    # COM should be (0,0)
    x_rotated -= np.mean(points_rotated[0, :])
    y_rotated -= np.mean(points_rotated[1, :])

    return x_rotated, y_rotated


def get_relative_pos(x : NDArray[Any]) -> NDArray[Any]:
    '''
    Get relative positions of each node based on the center of mass (0,0).

    Parameters
    -----
    x: nupmy array of positions in a given axis. 
        Shape : (num_timepoints, N)

    Returns
    -----
    rel_x: numpy array of positions in a given axis, relative to the center of
    mass at each time point.
        Shape: (num_timepoints, N)
    '''
    N = x.shape[1]
    com_x = np.nanmean(x, axis=1)                   # (num_timepoints,)
    com_x = np.repeat(com_x, N).reshape(x.shape)    # (num_timepoints, N)
    rel_x = x - com_x                               # (num_timepoints, N)

    return rel_x