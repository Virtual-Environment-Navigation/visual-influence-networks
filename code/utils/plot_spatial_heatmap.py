import numpy as np
from typing import Callable, Union, Literal
import matplotlib.pyplot as plt
from scipy import stats, optimize
import os

'''
Gaussian function (1d) gives:
    a: the height of the curve's peak
    b: the position of the center of the peak
    c: the width of the bell (standard deviation)
There would be 2 values of b & c for 2d gaussian.

Use this to plot spatial heatmaps in SD rather than in meters.
'''

## ===== local functions for gaussian fitting =====
# Taken from https://scipy-cookbook.readthedocs.io/items/FittingData.html
#         -> gaussian(), moments(), fitgaussian()

# used in fitgaussian()
def gaussian(height : float,
             center_x : float,
             center_y : float,
             width_x : float,
             width_y : float) -> Callable[[float, float], float]:
    '''
    Returns a Gaussian function with the given parameters.

    Parameters
    -----
    height : float
        The peak height of the Gaussian.
    center_x : float 
        The x-coordinate of the center.
    center_y : float
        The y-coordinate of the center.
    width_x : float
        The standard deviation along the x-axis.
    width_y : float
        The standard deviation along the y-axis.

    Returns
    -----
    Callable[[float, float], float] : function
        A function that takes x and y coordinates and returns the value of the 
        Gaussian function at that point.
    '''
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)


# used in fitgaussian()
def moments(data : np.ndarray) -> tuple:
    '''
    Returns the gaussian parameters (height, x, y, width_x, width_y) of a 2D 
    distribution by calculating its moments.

    Parameters
    -----
    data : numpy array of numbers
        Shape (num_x_cells, num_y_cells).
        A 2D array representing the data distribution.

    Returns
    -----
    tuple : (float, float, float, float, float)
        The Gaussian parameters: (height, x, y, width_x, width_y).
        See the parameter description in gaussian().
    '''
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()

    return (height, x, y, width_x, width_y)


# used in get_SD()
def fitgaussian(data : np.ndarray) -> tuple:
    '''
    Returns the gaussian parameters (height, x, y, width_x, width_y) of a 2D 
    distribution found by a fit.
    It uses least squares fitting to fit a Guassian.

    Parameters
    -----
    data : numpy array
        Shape (num_x_cells, num_y_cells).
        A 2D array representing the data distribution.

    Returns
    -----
    tuple : (float, float, float, float, float)
        The Gaussian parameters: (height, x, y, width_x, width_y).
        See the parameter description in gaussian().
    '''
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


## ===== get PDF & SD after gaussian fitting =====
# used in get_SD()
def get_PDF(pos_x : Union[np.ndarray, list], 
            pos_y : Union[np.ndarray, list], 
            xmin : int = None,
            xmax : int = None,
            ymin : int = None,
            ymax : int = None,
            grid_width : Union[int, float] = 0.5) -> np.ndarray:
    '''
    Fit 2d gaussian & get PDF.
    Refer to: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Parameters
    -----
    pos_x, pos_y : numpy array or python list
        Shape (num_datapoints,).
        x & y positions (in meters).

    [Optional]
    xmin, xmax, ymin, ymax : int
        Limits on x & y coordinates.
        default = int closest to the min/max in pos_x/pos_y.
    grid_width (default = 0.5) : float
        Defines cell size in the grid.

    Returns
    -----
    Z : numpy array
        Shape (num_x_cells, num_y_cells).
        Values representing PDF.
    '''
    # ----- convert python list to numpy array-----
    if isinstance(pos_x, list):
        pos_x = np.array(pos_x)
    if isinstance(pos_y, list):
        pos_y = np.array(pos_y)

    # ----- create grid positions -----
    xmin = int(np.floor(np.min(pos_x))) if xmin is None else xmin
    xmax = int(np.ceil(np.max(pos_x))) if xmax is None else xmax
    ymin = int(np.floor(np.min(pos_y))) if ymin is None else ymin
    ymax = int(np.ceil(np.max(pos_y))) if ymax is None else ymax
    X, Y = np.mgrid[xmin : xmax+grid_width : grid_width,
                    ymin : ymax+grid_width : grid_width]
    grid_XY = np.vstack([X.ravel(), Y.ravel()])

    # ----- formatting -----
    # remove all nan
    ind = ~np.isnan(pos_x) * ~np.isnan(pos_y)
    pos_x = pos_x[ind]
    pos_y = pos_y[ind]
    pos = np.vstack([pos_x, pos_y])  # (2, num_pos)

    # ----- get SD -----
    # perform a kernel density estimate (KDE) to estimate PDF
    # -> a smoothed version of the histogram of the data
    kernel = stats.gaussian_kde(pos)
    # calculate PDF at each grid point
    # Z : (num_cell_x, num_cell_y) - same shape as X & Y
    Z = np.reshape(kernel(grid_XY).T, X.shape)

    return Z


def get_SD(pos_x : Union[np.ndarray, list], 
           pos_y : Union[np.ndarray, list], 
           xmin : int = None,
           xmax : int = None,
           ymin : int = None,
           ymax : int = None,
           grid_width : Union[int, float] = 0.5
           ) -> tuple[float, float, np.ndarray]:
    '''
    Fit 2d gaussian & get SD.
    Refer to: 
    * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    * https://scipy-cookbook.readthedocs.io/items/FittingData.html 

    Parameters
    -----
    pos_x, pos_y : numpy array or python list
        Shape (num_datapoints,).
        x & y positions (in meters).

    [Optional]
    xmin, xmax, ymin, ymax : int
        Limits on x & y coordinates.
        default = integers closest to the min/max in pos_x/pos_y
    grid_width (float): default = 0.5
        Defines cell size in the grid.

    Returns
    -----
    SD_width_x, SD_width_y : float
        The distance 1 SD represents on the x & y axes (in meters)
    Z : numpy array
        Shape (num_x_cells, num_y_cells).
        Values representing PDF.
    '''
    # calculate PDF at each grid point
    # Z : (num_cell_x, num_cell_y) - same shape as X & Y
    Z = get_PDF(pos_x, pos_y,
                xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax,
                grid_width=grid_width)
    # fit a Gaussian distribution, using least squares, to the PDF to get SDs
    _, _, _, width_x, width_y = fitgaussian(Z)
    # Gaussian fitting is in units of grid_width
    SD_width_x = width_x * grid_width
    SD_width_y = width_y * grid_width

    return SD_width_x, SD_width_y, Z


# ===== transform the units based on SD from gaussian fitting =====
def transform_m2SD(pos_x_M : Union[np.ndarray, list], 
                   pos_y_M : Union[np.ndarray, list], 
                   SD_width_x : float, 
                   SD_width_y : float
                   ) -> tuple[np.ndarray, np.ndarray]:
    '''
    Transform given positions (m) to new positins (SD).

    Parameters
    -----
    pos_x_M, pos_y_M : numpy array or python list 
        Shape (num_datapoints,).
        x & y positions (in meters).
    SD_width_x, SD_width_y : float
        The distance 1 SD represents on the x & y axes (in meters).

    Returns
    -----
    pos_x_SD, pos_y_SD : numpy array of float
        Shape (num_datapoints,).
        x & y positions (in SD).
    '''
    if isinstance(pos_x_M, list):
        pos_x_M = np.array(pos_x_M)
    if isinstance(pos_y_M, list):
        pos_y_M = np.array(pos_y_M)
    
    pos_x_SD = pos_x_M / SD_width_x
    pos_y_SD = pos_y_M / SD_width_y

    return pos_x_SD, pos_y_SD


# ===== get a mask to filter out the data points =====
def get_filter(x : list, 
               y : list, 
               xmin : int = None,
               xmax : int = None,
               ymin : int = None,
               ymax : int = None,
               grid_width : Union[int, float] = 0.5,
               filter_method : Literal['cell_count', 'prob_density'] = 'cell_count',
               threshold : Union[int, float] = None) -> np.ndarray[np.bool_]:
    '''
    Get a mask to filter out the data points that are in cells where 
    probability (based on a given PDF) is less than a threshold & update PDF
    (set the probability of filtered cells to 0).

    Parameters
    -----
    x, y : python list
        Shape (num_datapoints,).
        x & y positions.

    [Optional]
    xmin, xmax, ymin, ymax : int
        Limits on x & y coordinates.
        default = integers closest to the min/max in pos_x/pos_y
    grid_width (default = 0.5) : float
    filtering method (default = 'cell_count') : str
        the outputs are removed if... 
        * if 'cell_count' -> # of datapoints in the cell is less than 
        a threshold (default = 5) 
        * if 'prob_density' -> cell probability is less than 
        a threshold (default = 0.001) 
    threshold : int or float
        threashold used to filter outlier datapoints out.
        * if 'cell_count' -> default = 5
        * if 'prob_density' -> default = 0.001

    Returns
    -----
    mask : numpy array of bool
        Shape (num_datapoints,)
        * if True -> the datapoint should remain
        * if False -> the datapoint should be filtered out
    '''
    # init
    xmin = int(np.floor(np.min(x))) if xmin is None else xmin
    xmax = int(np.ceil(np.max(x))) if xmax is None else xmax
    ymin = int(np.floor(np.min(y))) if ymin is None else ymin
    ymax = int(np.ceil(np.max(y))) if ymax is None else ymax

    if filter_method == 'cell_count':
        # init
        threshold = 5 if threshold is None else threshold
        x_range = np.arange(xmin, xmax + grid_width, grid_width)
        y_range = np.arange(ymin, ymax + grid_width, grid_width)
        # count data points per cell
        counts, _, _ = np.histogram2d(x, y, bins=[x_range, y_range])
        mask = np.zeros(len(x), dtype=bool)
        for ix in range(len(x_range) - 1):
            for iy in range(len(y_range) - 1):
                if counts[ix, iy] < threshold:
                    cell_mask = (x >= x_range[ix]) & (x < x_range[ix + 1]) & (y >= y_range[iy]) & (y < y_range[iy + 1])
                    mask |= cell_mask

        return ~mask

    elif filter_method == 'prob_density':
        threshold = 0.001 if threshold is None else threshold
        x_range = np.arange(xmin, xmax + grid_width, grid_width)
        y_range = np.arange(ymax, ymin - grid_width, -grid_width)
        # get cell indices where PDF < threshold & update PDF
        Z = get_PDF(x, y, 
                    xmin=xmin, xmax=xmax,
                    ymin=ymin, ymax=ymax,
                    grid_width=grid_width)
        low_prob_indices = np.where(Z < threshold)
        # create a mask to filter out data points from x & y
        mask = np.ones(len(x), dtype=bool)
        for i, j in zip(*low_prob_indices):
            # get cell indices for each datapoint (num_x_values = num_y_values)
            x_digitized = np.digitize(x, x_range)
            y_digitized = np.digitize(y, y_range)
            # find indices 
            indices = np.where((x_digitized == i) & (y_digitized == j))[0]
            mask[indices] = False

        return mask


## ===== plot heat maps =====
def plot_leadership_heatmap(leadership_values : np.ndarray,
                            xmin : int,
                            xmax : int,
                            ymin : int,
                            ymax : int,
                            unit : Literal['m', 'SD'],
                            cmap_name : str = 'jet',
                            grid_width : Union[int, float] = 0.5,
                            saved : bool = True,
                            title : str = None,
                            output_folder : str = '../output/heatmap/',
                            output_file : str = 'spatial_heatmap',
                            output_format : str = 'png') -> None:
    ''' 
    Plot mean leadership value for each cell in spatial heat map.

    Parameters
    -----
    leadership_values: numpy array of numbers
        Shape (num_x_cells, num_y_cells).
        Leadership values to be plotted in the heat map.
    xmin, xmax, ymin, ymax : int
        Limits on x & y coordinates.
        default = integers closest to the min/max in pos_x/pos_y
    unit : str ('m' or 'SD')
    
    [Optional]
    cmap_name (default : 'jet') : str
        colormap name (e.g., 'jet', 'gist_earth')
    grid_width (default = 0.5) : float
        cell size of the grid (in meters).
        each cell: grid_width x grid_width
    saved (default = True) : bool
        * if True -> save plot
        * if False -> show plot
    title (default) : str
        if not None, the string is added to the title
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
    # --- init ---
    x_range = np.arange(int(xmin), int(xmax)+1)
    y_range = np.arange(int(ymax), int(ymin-1), -1)

    # adjust grid_width so that 1/grid_width is an integer
    if grid_width <= 0:
        grid_width = 0.5     # default
    elif grid_width >= 1:
        grid_width = 1
    elif not (1/grid_width).is_integer():
        grid_width = 1 / round(1/grid_width)

    # --- plot ---
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap(cmap_name)
    # transpose as imshow needs a value of shape (num_rows, num_columns) 
    # = (num_y_cells, num_x_cells)
    im = ax.imshow(np.transpose(leadership_values), cmap=cmap)
    ax.set_xticks(np.arange(0, np.shape(leadership_values)[0], 1/grid_width), 
                  x_range)
    ax.set_yticks(np.arange(0, np.shape(leadership_values)[1], 1/grid_width), 
                  y_range)
    ax.set_xlabel('Left-Right (' + unit + ')', fontsize=15)
    ax.set_ylabel('Back-Front (' + unit + ')', fontsize=15)
    fig.colorbar(im)
    if title is not None:
        plt.title(title, fontsize=18)
        # plt.title("leadership spatial heat map: " + output_file)

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

def plot_PDF(PDF : np.ndarray,
             xmin : int, 
             xmax : int, 
             ymin : int, 
             ymax : int,
             unit : Literal['m', 'SD'],
             cmap_name : str = 'jet',
             grid_width : Union[int, float] = 0.5,
             saved : bool = True,
             plotGaussian : bool = False, 
             title : str = None,
             output_folder : str = 'output/PDF/',
             output_file : str = 'PDF',
             output_format : str = 'png') -> None:
    '''
    Plot the PDF. Refer to: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

    Parameters
    -----
    PDF : numpy array
        Shape (num_x_cells, num_y_cells).
        probability densities to be plotted.
    xmin, xmax, ymin, ymax : int
        Limits on x & y coordinates.
        default = integers closest to the min/max in pos_x/pos_y
    unit : str ('m' or 'SD')
    
    [Optional]
    cmap_name (default : 'jet') : str
        colormap name (e.g., 'jet', 'gist_earth')
    grid_width (default = 0.5) : float
        cell size of the grid (in meters).
        each cell: grid_width x grid_width
    saved (default = True) : bool
        * if True -> save plot
        * if False -> show plot
    plotGaussian (default = False) : bool
        * if True -> plot Gaussian lines
        * if False -> don't plot Gaussian lines
    title (default = None) : str
        if not None, the string is added to the title
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
    # --- error handling ---
    if unit!='m' and unit!='SD':
        raise ValueError("Wrong unit; please set it to 'm' or 'SD'")
    # adjust grid_width so that 1/grid_width is an integer
    if grid_width <= 0:
        grid_width = 0.5     # default
    elif grid_width >= 1:
        grid_width = 1
    elif not (1/grid_width).is_integer():
        grid_width = 1 / round(1/grid_width)

    # --- init ---
    x_range = np.arange(int(xmin), int(xmax)+1)
    y_range = np.arange(int(ymax), int(ymin-1), -1)

    # --- plot ---
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap(cmap_name)   # jet, gist_earth
    im = ax.imshow(np.rot90(PDF), cmap=cmap)

    if plotGaussian:
        params = fitgaussian(PDF) 
        fit = gaussian(*params)
        (height, x, y, width_x, width_y) = params   # CAREFUL: in cell #s (not in m)
        ax.contour(np.rot90(fit(*np.indices(PDF.shape))), colors='black')
        plt.text(0.95, 0.05, """
        width_x : %.2f
        width_y : %.2f""" %(width_x*0.5, width_y*0.5),
                fontsize=10, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes,
                c="white")
        
    ax.set_xticks(np.arange(0, np.shape(PDF)[0], 1/grid_width), 
                  x_range)
    ax.set_yticks(np.arange(0, np.shape(PDF)[1], 1/grid_width), 
                  y_range)
    ax.set_xlabel('Left-Right ('+unit+')', fontsize=15)
    ax.set_ylabel('Back-Front ('+unit+')', fontsize=15)
    fig.colorbar(im)
    if title is not None:
        plt.title(title, fontsize=18)

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
