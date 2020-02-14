import numpy as np

def angular_distance(l1,b1,l2,b2,deg=True):
    """
    Calculate angular distance on a sphere from longitude/latitude pairs to other using Great circles
    in units of deg
    :param: l1    longitude of point 1 (or several)
    :param: b1    latitude of point 1 (or several)
    :param: l2    longitude of point 2
    :param: b2    latitude of point 2
    :option: deg  option carried over to GreatCricle routine
    """
    # calculate the Great Circle between the two points
    # this is a geodesic on a sphere and describes the shortest distance
    gc = GreatCircle(l1,b1,l2,b2,deg=deg)
    
    # check for exceptions
    if gc.size == 1:
        if gc > 1:
            gc = 1.
    else:
        gc[np.where(gc > 1)] = 1.

    return np.rad2deg(np.arccos(gc))    
    
    
def GreatCircle(l1,b1,l2,b2,deg=True):
    """
    Calculate the Great Circle length on a sphere from longitude/latitude pairs to others
    in units of rad on a unit sphere
    :param: l1    longitude of point 1 (or several)
    :param: b1    latitude of point 1 (or several)
    :param: l2    longitude of point 2
    :param: b2    latitude of point 2
    :option: deg  Default True to convert degree input to radians for trigonometric function use
                  If False, radian input is assumed
    """
    if deg == True:
        l1,b1,l2,b2 = np.deg2rad(l1),np.deg2rad(b1),np.deg2rad(l2),np.deg2rad(b2)

    return np.sin(b1)*np.sin(b2) + np.cos(b1)*np.cos(b2)*np.cos(l1-l2)    
