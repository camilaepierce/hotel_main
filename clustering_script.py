import sys
import numpy as np
import scraping_script
from library import cubic_spline, scrape_google_maps, visualize_clusters
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    if len(sys.argv) == 4:
        snapped_path = sys.argv[1]
        hotels_path = sys.argv[2]
        name = sys.argv[3]

    ### Tunable Parameters
    ### Number of clusters
    k = 4
    ### Miles from highway to search for hotels
    n=3

    ### Beta method ("scaling", "adding", "bool_value")
    init_method = "adding"

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

    # ### Visualize both clusterings
    r4_info = visualize_clusters.visualize_clusters(scaled_vectors_r4, kmeans_r4, k, title=f"{name} R^4 Control", plot_rating=False)
    # Plot highway
    visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
    plt.savefig(f"results/{name}_with_signed_distance")
    plt.show(block=False)

    r3_info = visualize_clusters.visualize_clusters(scaled_vectors_r3, kmeans_r3, k, title=f"{name} R^3 Control", plot_rating=False)
    # Plot highway
    visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
    plt.savefig(f"results/I-{name}_Control_without_signed_distance")
    plt.show(block=False)

    ### Compare created clusters
    print("##############################################")
    print("Data Comparisons")
    print("With Signed Distance\t\tWithout Signed Distance")
    print(f"Inertia:\t\t{r4_info["inertia"]}\t\t{r3_info["inertia"]}")
    print("By Cluster:")
    for label in range(k):
        print(f"Cluster {label}")
        print(f"Area:\t\t{r4_info["clusters"][label]["area"]}\t\t{r3_info["clusters"][label]["area"]}")
        print(f"Rating Average:\t\t{r4_info["clusters"][label]["cluster_center"][2]}\t\t{r3_info["clusters"][label]["cluster_center"][2]}")
        print(f"Number of Members:\t\t{len(r4_info["clusters"][label]["members"])}\t\t{len(r3_info["clusters"][label]["members"])}")
        print(f"Distance Average:\t\t{r4_info["clusters"][label]["cluster_center"][3]}")
    input("Press enter to close windows comparison.")


    ###To-Do: update title info, include legend with hull areas, average ratings
    ### change beta values


    ##############################################################
    ### Manually change Beta value ( * signed distance)

    for current_method in ["scaling", "adding", "bool_value"]:
        visualize_clusters.increment_beta_values(scaled_vectors_r4, hw_cubic_spline_scaled, name, k, n, method=current_method)

    input("Press enter to end program.")

    ### Find more precise determination of beta - iteration, minimize
    ### Create method to combine both distance from a highway and side of the highway the hotels (?) are on
    ### Create a dataset that mirrors real life, using highway + hotel data, with ratings - google maps, trip advisor
    ### Display third feature of direction + distance to highway
    ### simplify above code, modularize, figure out precise data collection (?)
    ### Find method for correctness / accuracy

    # weighting function or weighting parameter, tune parameter (manually?)
    # come up with different ways to tune, higher level cost function
    #   *** tune visually, not based on inertia or overlap area?
