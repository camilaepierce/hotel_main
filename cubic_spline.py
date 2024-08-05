"""
Created by Camila Pierce
Last Updated 8.5.2024

Set of functions to process and modify data related to highway distance.

Reformatted using black.
"""
import numpy as np
from scipy.optimize import minimize
import math

def reformat_data(data, highway_spline, decimals = 4):
    """
    Takes data of at least x, y and adds smallest distance of each point to highway with given coefficients.

    Parameters:
    * data (numpy array) :
    * highway_spline (CubicSpline) :
    * decimals (int) : number of decimals to display data to

    Returns:
        Copy of numpy array with shortest distance to highway added for each point.
    """
    r_three = []
    for data_point in data:
        r_three.append([create_third_dimension(highway_spline, data_point)])
    np.set_printoptions(suppress=True, precision = decimals)
    return np.append(data, r_three, axis=1)



def find_minimum_distance(spline_est, full_point):
    """
    Returns x, y, distance of closest point on a highway.
    """
    point = (full_point[0], full_point[1])
    new_min = minimize(updated_distance_fxn(spline_est, point), point[0])
    all_guesses = new_min.x
    if len(all_guesses) == 0:
        print("No minimums")
        return
    if len(all_guesses) == 1:
        xi = all_guesses[0]
        # print((xi, float(spline_est(xi))))
        # print(point)
        return xi, float(spline_est(xi)), math.dist((xi, float(spline_est(xi))), point)

    xi  = all_guesses[0]
    final_min = math.dist((xi, spline_est(xi)), point)
    for x in all_guesses:
        if math.dist((x, spline_est(x)), point) < final_min:
         xi = x
    return xi, float(spline_est(xi)), final_min

def updated_distance_fxn(spline, point):
    def helper(xi):
        if not isinstance(xi, (np.ndarray, list, tuple)):
            return math.dist((xi, spline(xi)), point)
        else:
            out = []
            for x in xi:
                out.append(math.dist((x, spline(x)), point))
            return out
    return helper

def signed_RHS_LHS_spline(point, spline, hw_x, hw_y, distance):
    """
        Uses cross product to determine if point is on the RHS or LHS of a given highway.
    """
    x = [1, spline.derivative()(hw_x), 0]
    y = [point[0] - hw_x, point[1] - hw_y, 0]
    c = np.cross(x, y)
    cross_product = c[2]
    if cross_product > 0:
        return distance
    return -distance

def create_third_dimension(spline, point):
    x, y, min_dist = find_minimum_distance(spline, point)
    return signed_RHS_LHS_spline(point, spline, x, y, min_dist)


def retrieve_hotel_data(file_name):
    """ Reads txt file that contains hotel data.
        Assumes in columns of Name, X location, Y location, and Rating."""
    file = open(file_name, "r")
    file.readline()
    data = []
    for line in file.readlines():
        data.append(np.array([float(d) for d in line.split()[1:4]]))
    return np.array(data)
