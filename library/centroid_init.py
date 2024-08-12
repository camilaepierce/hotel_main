import numpy as np
import math

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
