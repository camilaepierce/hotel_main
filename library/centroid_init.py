import numpy as np
import math
from scipy.integrate import quad
import numpy.polynomial.polynomial as np_poly
from scipy.optimize import minimize

def find_centroid(data, dimension_indexes = (0, 1)):

    centroid = []
    for dimension in dimension_indexes:
        centroid.append(np.mean(data[:, dimension]))
    return centroid


def mod_data(data, dimension_indexes=(0, 1)):
    return data[:, dimension_indexes]

def find_closest_point(modded_data, centroid):
    min_distance = float("inf")
    min_index = -1

    for ix, point in enumerate(modded_data):
        dist = math.dist(centroid, point)
        if dist < min_distance:
            min_distance = dist
            min_index = ix
    return min_index

# def find_furthest_point(modded_data, c_list):
#     max_distance = float("-inf")
#     max_index = -1

#     for ix, point in enumerate(modded_data):
#         dist = sum([math.dist(c_x, point) for c_x in c_list])
#         if dist >  max_distance:
#              max_distance = dist
#              max_index = ix
#     return  max_index

def find_furthest_point(modded_data, c_list):
    max_distance = float("-inf")
    max_index = -1

    for ix, point in enumerate(modded_data):
        dist = min([math.dist(c_x, point) for c_x in c_list])
        if dist >  max_distance:
             max_distance = dist
             max_index = ix
    return  max_index


def create_init_vectors(r_x_data, dims, k):
    center = find_centroid(r_x_data, dims)
    r_4_modded = mod_data(r_x_data, dims)
    first_index = find_closest_point(r_4_modded, center)
    n = [first_index]
    while len(n) < k:
        next_index = find_furthest_point(r_4_modded, [r_4_modded[i] for i in n])
        n.append(next_index)
    return [r_x_data[i] for i in n] #returns list of initialization vectors

def reshape_cubic_coeff(array):
    new_shape = array.shape[::-1]
    flat = array.flatten(order = "F")
    return np.reshape(flat, new_shape)


def arc_length_div_init(cubic_spline, k, data):
    total_length=0
    #reformat coeff
    cubic_coeff = reshape_cubic_coeff(cubic_spline.c)

    print(len(cubic_spline.x))
    print(len(cubic_spline.c))
    print(cubic_spline.x)
    print(cubic_spline.c)
    print(cubic_coeff)
    for i in range(len(cubic_spline.x)-1):
        a = cubic_spline.x[i]
        b = cubic_spline.x[i+1]
        h = h_of_x(cubic_coeff[i])#simplification of arc length funciton
        total_length += quad(h, a, b)[0]

    unit_length = total_length / (k-1)
    x_values = []
    for i in range(k):
        goal_arc_length = i * unit_length
        goal_fxn = arc_length_goal_fxn(cubic_spline, cubic_coeff, goal_arc_length)
        result = minimize(goal_fxn, (cubic_spline.x[0]+goal_arc_length)).x
        if len(result) > 1:
            print("multiple results yielded")
        else:
            x_values.extend(result)

    print("expected first and last:", cubic_spline.x[0], cubic_spline.x[-1])
    print("final x values", x_values)

    modded_data = mod_data(data, (0, 1))
    init_vectors = []
    for x in x_values:
        current_point = (x, cubic_spline(x))
        init_vectors.append(data[find_closest_point(modded_data, current_point)])

    return np.array(init_vectors)


    #.x breakpoints .c coefficients

def h_of_x(cubic_coeff):
    """
    sqrt(1 + f'(x)^2)
    """
    p = np_poly.polyder(cubic_coeff)#derive segment of cubic spline
    sq_p = np_poly.polypow(p, 2)#square
    def special_fxn(x):
        y = math.sqrt(1 + (np.polynomial.polynomial.Polynomial(sq_p)(x)))
        return y
    return special_fxn

def arc_length_goal_fxn(cubic_spline, coeffs, goal_arc_length):
    #create cubic spline specific info?
    def special_fxn(x):
        #pre-integrated
        #go through segements until x < start of next segement.
        #raise error if before first breakpoint
        #total length as passing through until desired segment
        #from there, find until just x, add to total
        #return total
        if x < cubic_spline.x[0]:
            return float("inf")#
        length = 0
        last_bp = cubic_spline.x[-1]
        for i in range(len(cubic_spline.x)-1):
            a = cubic_spline.x[i]
            b = cubic_spline.x[i+1]
            coeff = coeffs[i]
            if x < b:
                return (length + quad(h_of_x(coeff), a, x)[0] - goal_arc_length)**2
            else:
                length += quad(h_of_x(coeff), a, b)[0]
        return (length + quad(h_of_x(coeff), last_bp, x)[0] - goal_arc_length)**2

    return special_fxn
