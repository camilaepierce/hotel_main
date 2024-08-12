"""
clustering_script.py
Created by Camila Pierce
Last Updated 8.7.2024

Run either in the command line or by clicking run on file.

If in command line, include the following:
python clustering_script.py [snapped path] [hotel data] [name]
**Example command line test:
python clustering_script.py .\snapped_highways\test_input_path.txt .\hotel_data\test_input_hotels.txt test_input

If hitting run on file, ensure the correct paths for lines 41-43.

Tunable hyper-parameters are included on lines 47-56. Various visualizations are avaiable depending on test to run
"""

import sys
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from library.centroid_init import create_init_vectors

import scraping_script
from library import cubic_spline, scrape_google_maps, visualize_clusters

if __name__ == "__main__":
    import os

    if len(sys.argv) == 4:
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        snapped_path = sys.argv[1]
        hotels_path = sys.argv[2]
        name = sys.argv[3]
    elif len(sys.argv) == 1:
        ### FILE PATHS HERE
        cwd = os.getcwd()  # Get the current working directory (cwd)
        files = os.listdir(cwd)  # Get all the files in that directory
        snapped_path = ".\\snapped_highways\\test_input_path.txt"
        hotels_path = ".\\hotel_data\\test_input_hotels.txt"
        name = "test_input"

    ### Tunable Parameters
    ### Number of clusters
    k = 4
    ### Miles from highway to search for hotels
    n=3
    ### Beta method ("scaling", "adding", "bool_value")
    beta_method_list = ["scaling", "adding", "bool_value"]

    ### Beta matrix
    beta_array = ((0.5, 1.0),
                   (1.5, 2.0),
                   (3.0, 5.0))


    ######################################
    ### Retrieving Data ###
    ######################################

    hotel_vectors_r3 = cubic_spline.retrieve_hotel_data(hotels_path)
    true_path = scraping_script.read_data(snapped_path)

    scaled_vectors_r3 = StandardScaler().fit(hotel_vectors_r3).transform(hotel_vectors_r3)


    hw_cubic_spline, hw_cubic_spline_scaled = cubic_spline.create_cubic_splines(true_path)


    ######################################
    ### Reformatting Data ###
    ######################################

    ### Add signed distance to highway
    hotel_vectors_r4 = cubic_spline.reformat_data(hotel_vectors_r3, hw_cubic_spline)
    scaled_vectors_r4 = StandardScaler().fit(hotel_vectors_r4).transform(hotel_vectors_r4)

    #####################################
    ### Clustering ###
    #####################################


    ### Run kmeans++ algorithm on R^4
    kmeans_r4 = KMeans(n_clusters=k, init='k-means++')
    estimator_r4 = kmeans_r4.fit(scaled_vectors_r4)

    ### Run kmeans++ algorithm on R^3
    kmeans_r3 = KMeans(n_clusters=k, init='k-means++')
    estimator_r3 = kmeans_r3.fit(scaled_vectors_r3)

    ######################################
    ### Interpret ###
    ######################################

    # # ### Visualize both clusterings
    # r4_info = visualize_clusters.visualize_clusters(scaled_vectors_r4, kmeans_r4, k, title=f"{name} R^4 Control", plot_rating=False)
    # # Plot highway
    # visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
    # plt.savefig(f"results/{name}_with_signed_distance")
    # plt.show(block=False)

    # r3_info = visualize_clusters.visualize_clusters(scaled_vectors_r3, kmeans_r3, k, title=f"{name} R^3 Control", plot_rating=False)
    # # Plot highway
    # visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
    # plt.savefig(f"results/I-{name}_Control_without_signed_distance")
    # plt.show(block=False)

    # ### Compare created clusters
    # print("##############################################")
    # print("Data Comparisons")
    # print("With Signed Distance\t\tWithout Signed Distance")
    # print(f"Inertia:\t\t{r4_info["inertia"]}\t\t{r3_info["inertia"]}")
    # print("By Cluster:")
    # for label in range(k):
    #     print(f"Cluster {label}")
    #     print(f"Area:\t\t{r4_info["clusters"][label]["area"]}\t\t{r3_info["clusters"][label]["area"]}")
    #     print(f"Rating Average:\t\t{r4_info["clusters"][label]["cluster_center"][2]}\t\t{r3_info["clusters"][label]["cluster_center"][2]}")
    #     print(f"Number of Members:\t\t{len(r4_info["clusters"][label]["members"])}\t\t{len(r3_info["clusters"][label]["members"])}")
    #     print(f"Distance Average:\t\t{r4_info["clusters"][label]["cluster_center"][3]}")
    # input("Press enter to close windows comparison.")


    ###To-Do: update title info, include legend with hull areas, average ratings
    ### change beta values


    ##############################################################
    #check variability for greedy, R^4,
    #vanilla initialization
    # visualize_clusters.temp_variability_demo(scaled_vectors_r3, hw_cubic_spline_scaled, title="repeated_control_r3_vanilla")

    #check variability for greed vs vanilla, centroid initialization in R^2, R^3, R^4
    # init_vectors = create_init_vectors(scaled_vectors_r4, (0, 1, 2, 3), k)
    # visualize_clusters.temp_variability_demo(scaled_vectors_r4, hw_cubic_spline_scaled, title="repeated_control_r4_greedy_4dim_init",
    #                                          special_init=init_vectors, supertitle=f"Repeated Control, R^4 Data, Lat Long Rating Sign Initialization")

    # visualize_clusters.temp_variability_demo(scaled_vectors_r4, hw_cubic_spline_scaled, title="repeated_control_r4_greedy")
    # visualize_clusters.temp_variability_demo(scaled_vectors_r3, hw_cubic_spline_scaled, title="repeated_control_r3_greedy")
    # visualize_clusters.temp_variability_demo(scaled_vectors_r4, hw_cubic_spline_scaled, title="repeated_control_r4_vanilla")
    ## Manually change Beta value ( * signed distance)
    for beta_method in beta_method_list:
        visualize_clusters.increment_beta_values(scaled_vectors_r4, hw_cubic_spline_scaled, name,
                                                    k, n, method=beta_method, final_beta_list=beta_array)

    # input("Press enter to end program.")
