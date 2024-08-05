"""
Created by Camila Pierce
Last Updated 8.5.2024

visualize_clusters contains several functions that process and plot data after scipy kmeans
clustering. Mildly specific to hotels near highways. Most input data is expected to be in R^3
or R^4: x_location, y_location, rating, (signed distance): though rating and signed distance
may be other attributes.

Reformatted using black.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely import intersection, Polygon, get_coordinates
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# color map for visualization
COLORS = {
    0: "slateblue",
    1: "turquoise",
    2: "orange",
    3: "lawngreen",
    4: "darkviolet",
    5: "firebrick",
    6: "olivedrab",
    7: "goldenrod",
    8: "dodgerblue",
}


def visualize_clusters(
    data_vectors,
    kmeans_object,
    num_clusters,
    plot_rating=False,
    cmap_index=2,
    title="Untitled",
):
    """
    Plots data vectors into assigned clusters from running scipy kmeans.
    Does not show or save plot automatically.

    Parameters:
    * data_vectors (numpy array) :
    * kmeans_object (scipy KMeans) :
    * num_clusters (int) :

    Returns:
        Dictionary of cluster info {num : (int), inertia : (float),
        clusters: {label : (int),
            cluster_center : (numpy array), area : (float),
            convex_hull : (ConvexHull), members : (numpy_array)}
    """

    # Dictionary containing master information about clustering
    cluster_info = {
        "num": num_clusters,
        "inertia": kmeans_object.inertia_,
        "clusters": [
            ### Will contain a dictionary for each cluster with keys:
            # "label" - label assigned by kmeans
            # "cluster_center" - average of all cluster members - all dimensions
            # "area" - area of convex hull
            # "convex_hull" - ConvexHull object
            # "members" - array, data_vectors of each cluster
        ],
    }

    # Find locations of created clusters
    center_locations = kmeans_object.cluster_centers_

    # Prep cluster_members
    # Each entry in dictionary collects the indexes of data points belonging to that cluster
    cluster_members = {i: [] for i in range(num_clusters)}
    cluster_info["clusters"].extend(
        [
            {"label": i, "cluster_center": kmeans_object.cluster_centers_[i]}
            for i in range(num_clusters)
        ]
    )
    # Sorts indexes of data by cluster
    for index, label in enumerate(kmeans_object.labels_):
        cluster_members[label].append(index)

    convex_hull_borders = []

    fig, ax = plt.subplots()

    ### Geographic space
    for cluster in cluster_members:
        # Collects x and y of data points in each cluster to scatter plot correctly
        # starts building convex hull
        temp_x, temp_y, ch_format, full_member = [], [], [], []
        for member in cluster_members[cluster]:
            temp_x.append(data_vectors[member, 0])
            temp_y.append(data_vectors[member, 1])
            ch_format.append(
                np.array([data_vectors[member, 0], data_vectors[member, 1]])
            )
            full_member.append(data_vectors[member])
        plt.scatter(temp_x, temp_y, c=COLORS[cluster])
        # Creates convex hull for each cluster, saves vertices of each hull to convex_hull_borders
        # *Clusters of less than three data points cannot create a viable convex hull,
        # would throw error in ConvexHull module
        ch_format = np.array(ch_format)
        if len(ch_format) > 2:
            hull = ConvexHull(ch_format)
            cluster_info["clusters"][cluster]["convex_hull"] = hull
            cluster_info["clusters"][cluster]["area"] = hull.area
            # plots convex hull lines for this cluster
            for simplex in hull.simplices:
                plt.plot(
                    ch_format[simplex, 0], ch_format[simplex, 1], c=COLORS[cluster]
                )
            plt.scatter(center_locations[:, 0], center_locations[:, 1], c="red")
            convex_hull_borders.append([ch_format[index] for index in hull.vertices])
        else:
            cluster_info["clusters"][cluster]["area"] = 0
        cluster_info["clusters"][cluster]["members"] = np.asarray(full_member)
        cluster_info["clusters"][cluster]
        # Adjust text location to make visible ***
        plt.text(
            1.32,
            0.95 - cluster * 0.7,
            f"""Label: {cluster_info["clusters"][cluster]["label"]} |
                 Num Members: {len(cluster_info["clusters"][cluster]["members"])}\n
                 Area: {cluster_info["clusters"][cluster]["area"]:.4f}\n
                 Avg. Rating: {cluster_info["clusters"][cluster]["cluster_center"][2]:.2f}""",
            c=COLORS[cluster],
        )
        all_intersection_combos(convex_hull_borders)

    plt.subplots_adjust(right=0.65)
    fig.set_figwidth(8)
    # Plot after removing next five lines ***
    xlimit = plt.xlim()
    ylimit = plt.ylim()

    plt.xlim(xlimit)
    plt.ylim(ylimit)

    if (
        plot_rating
    ):  # plots rating gradient over existing data --interactive, hover? ***
        plt.scatter(
            data_vectors[:, 0],
            data_vectors[:, 1],
            cmap="viridis",
            c=data_vectors[:, cmap_index],
        )
        plt.colorbar(label="hotel rating")

    plt.title(label=title)
    return cluster_info


### Computing areas


def intersection_area(points_1, points_2, plot=True):
    """
    Computes the area of intersection of two convex hulls. If no intersection, return zero.

    Parameters:
    * points_1 (list) : list of x and y values of first convex hull
    * points_2 (list) : list of x and y values of second convex hull
    * plot (Bool) : if true, shades intersection area of convex hulls

    Returns:
        Area of intersection of given convex hulls.
    """
    poly1, poly2 = Polygon(points_1), Polygon(points_2)
    # finds intersection / overlap area if there is any
    isIntersection = intersection(poly1, poly2)
    if plot:
        points = get_coordinates(isIntersection)
        x = [i[0] for i in points]
        y = [i[1] for i in points]
        plt.fill(x, y, c="lightcoral", alpha=0.5)
    return Polygon(isIntersection).area


def all_intersection_combos(list_of_convex_borders, factor=1, plot=False):
    """
    Finds intersection area between every combination of two clusters.

    Parameters:
    * list_of_convex_borders (list) : list of convex hull vertex points
    * factor (int) : scaling factor

    Returns:
        Total overlap area of convex hulls multiplied by optional scaling factor
    """
    total_overlap_area = 0
    # print("Intersection Areas:")
    for index_1, cluster_1 in enumerate(list_of_convex_borders):
        for index_2, cluster_2 in enumerate(list_of_convex_borders):
            if index_1 < index_2:
                area = intersection_area(cluster_1, cluster_2, plot=plot)
                # print(f"Clusters {index_1} and {index_2}: {area:.8f}")
                total_overlap_area += area
    return total_overlap_area * factor


def create_convex_hulls(data, labels, plot=False, indexes=(0, 1)):
    """
    Generalized version of visualize_clusters. Not in use, see above for more details.
    """
    print("create_convex_hulls is deprecated")
    cluster_members = {i: [] for i in range(4)}

    for index, label in enumerate(labels):
        cluster_members[label].append(index)

    convex_hull_borders = []
    total_area = 0

    ### Geographic space
    for cluster in cluster_members:
        temp_x, temp_y, ch_format = [], [], []
        for member in cluster_members[cluster]:
            if plot:
                temp_x.append(data[member, indexes[0]])
                temp_y.append(data[member, indexes[1]])
            ch_format.append(
                np.array([data[member, indexes[0]], data[member, indexes[1]]])
            )
        ch_format = np.array(ch_format)
        hull = ConvexHull(ch_format)
        total_area += hull.area

        if plot:
            plt.scatter(temp_x, temp_y, c=COLORS[cluster])
            for simplex in hull.simplices:
                plt.plot(
                    ch_format[simplex, 0], ch_format[simplex, 1], c=COLORS[cluster]
                )
            # plt.scatter(centers[:, 0], centers[:, 1], c="red")
        convex_hull_borders.append([ch_format[index] for index in hull.vertices])
    return convex_hull_borders, total_area


def plot_attributes(
    data_vectors,
    cluster_members,
    title,
    indexes=(0, 1),
    attribute="Geographic",
    create_hulls=True,
    show_plot=False,
    save_plot=True,
):
    """
    Plots desired data, defaults are for geographic subspace, creating convex hulls.
    """
    print("plot_attribute is deprecated")
    ### Geographic space
    for cluster in cluster_members:
        temp_x, temp_y, ch_format = [], [], []
        for member in cluster_members[cluster]:
            temp_x.append(data_vectors[member, indexes[0]])
            temp_y.append(data_vectors[member, indexes[1]])
            if create_hulls:
                ch_format.append(
                    np.array(
                        [
                            data_vectors[member, indexes[0]],
                            data_vectors[member, indexes[1]],
                        ]
                    )
                )
        plt.scatter(temp_x, temp_y, c=COLORS[cluster])
        if create_hulls:
            ch_format = np.array(ch_format)
            hull = ConvexHull(ch_format)
            for simplex in hull.simplices:
                plt.plot(
                    ch_format[simplex, 0], ch_format[simplex, 1], c=COLORS[cluster]
                )
        # plt.scatter(centers[:, indexes[0]], centers[:, indexes[1]], c="red")
    plt.title(label=f"{title} | {attribute}")


def plot_cubic_spline_highway(spline_obj, num_samples=100):
    """
    Plots highway from cubic spline.

    Parameters:
    * spline_obj (CubicSpline) : cubic spline representation of highway
    * num_samples (int) : number of sample points to visualize highway
    """
    hw_x = np.linspace(spline_obj.x[0], spline_obj.x[-1], num=num_samples)
    plotting_fxn = spline_obj(hw_x)
    plt.plot(hw_x, plotting_fxn)


##################################################################
### Changing Beta values, determine best and re-run test with new value


def modify_data(data, b, mthd):
    """
    Adjust data to change signed distance weighting. Assumes data is R^4.

    Parameters:
    * data (numpy array) : array of data values to modify
    * b (float) : beta value to change data by
    * mthd (str) : method to adjust data
        -scaling: multiplies signed distance by beta value
        -adding: adds/subtracts beta value to/from +/- distance
        -bool_value: 0 if on LHS, beta if on RHS

    Returns:
        Modified data, numpy array
    """
    if mthd == "scaling":
        return np.array(
            [
                np.array((data[i, 0], data[i, 1], data[i, 2], data[i, 3] * b))
                for i in range(len(data))
            ]
        )
    elif mthd == "adding":
        return np.array(
            [
                np.array(
                    (
                        data[i, 0],
                        data[i, 1],
                        data[i, 2],
                        (data[i, 3] + b if data[i, 3] > 0 else data[i, 3] - b),
                    )
                )
                for i in range(len(data))
            ]
        )
    elif mthd == "bool_value":
        return np.array(
            [
                np.array(
                    (data[i, 0], data[i, 1], data[i, 2], (0 if data[i, 3] > 0 else b))
                )
                for i in range(len(data))
            ]
        )
    else:
        raise ValueError


def increment_beta_values(
    data_vectors,
    highway_cubic_spline,
    name,
    k,
    n,
    method="scaling",
    start=0.1,
    stop=5,
    increment=0.05,
):
    """
    Runs k-means of every beta modification from start to stop.
    Plots sklearn.adjusted_rand_score of clusters
    based on expected left/right distributions.

    User inputs beta values to be inspected, reruns clustering with given values.

    Parameters:
    * data_vectors (numpy array) : hotel data vectors
    * highway_cubic_spline (CubicSpline) : cubic spline representation of highway
    * name (str) : name of highway
    * k (int) : number of clusters
    * n (float) : hotels within this range from highway in miles
    * method (str) : method to modify data
    * start (float) : starting beta value
    * stop (float) : ending beta value
    * increment (float) : increment value between trials
    """

    ### incrementing through beta values
    beta = start
    all_betas = []
    beta_yields = []
    left_right_labels = [
        (0 if data_vectors[i, 3] > 0 else 1) for i in range(len(data_vectors))
    ]

    print("Beta\t Inertia\t Overlap Area\t Combined")
    while beta <= stop:
        ###modify initial data with beta value
        modified_data = modify_data(data_vectors, beta, mthd=method)
        ###run kmeans on modified data
        kmeans = KMeans(n_clusters=4, init="k-means++")
        estimator = kmeans.fit(modified_data)
        ###gather beta value with score
        all_betas.append(beta)
        beta_yields.append(adjusted_rand_score(left_right_labels, kmeans.labels_))
        beta += increment

    #### plot data evaluation with beta values
    plt.figure()
    plt.plot(all_betas, beta_yields, "o-r")
    plt.show(block=False)

    # collect floats for which to rerun clustering
    final_beta_list = [
        float(beta)
        for beta in (
            input("Optimal Beta value (floats separated by a space or single float): ")
        ).split()
    ]
    for final_beta in final_beta_list:
        modified_data = modify_data(data_vectors, final_beta, method)
        ###run kmeans on modified data
        kmeans = KMeans(n_clusters=k, init="k-means++")
        estimator = kmeans.fit(modified_data)

        visualize_clusters(modified_data, kmeans, k, plot_rating=True,
            title=f"Beta {final_beta} of {name} with {k} clusters within {n} miles",
        )
        plot_cubic_spline_highway(highway_cubic_spline)

        plt.savefig(
            f"hotel_testing_betas_{method}/I-{name} \
                Hotel Clustering with Signed Distance Beta_{final_beta}.png"
        )###IF ERROR CHECK HERE FIRST
        plt.show(block=False)


#################################################################
### Testing ###
#################################################################

# geo_x = [0.0, 0.6, -0.7, 0.1]
# geo_y = [0.7, -0.4, 0.0, -0.6]
# att_x = [0.2, 0.5, 0.8, 1.4]
# att_y = [1.4, 1.0, 0.6, 0.3]
# centers_list = list(zip(geo_x, geo_y, att_x, att_y))

# data_vectors, data_labels, centers = make_blobs(n_features = 4,
# n_samples = 40, cluster_std=0.35, centers=centers_list, return_centers=True)

# kmeans = KMeans(n_clusters=4, init='k-means++')
# estimator = kmeans.fit(data_vectors)


# info = visualize_clusters(data_vectors, kmeans, estimator, 4, plot_rating=False)
# print(info)
# plt.show()

# points = [(35.828147, -78.682148), (35.835497, -78.668780), (35.836619, -78.656646),
# (35.835227, -78.642913), (35.832792, -78.630725), (35.821936, -78.612958),
# (35.811913, -78.603259)]
# out, lat, long = lat_long_snapped_path(points)


# spline = CubicSpline(long, lat)

# hw_x = np.linspace(long[0], long[-1], num=100)
# plotting_fxn = spline(hw_x)

# plt.plot(hw_x, plotting_fxn)
# plt.savefig("Highway Only")
# plt.show(block=False)
