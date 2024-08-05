import numpy as np
import cubic_spline
import scrape_google_maps
import googlemaps
import visualize_clusters
from scipy.interpolate import CubicSpline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

######################################
### Collecting Data ###
######################################

######### Test Modificaiton
### 440
# name = "440"
# points = [(35.828147, -78.682148), (35.835497, -78.668780), (35.836619, -78.656646), (35.835227, -78.642913), (35.832792, -78.630725), (35.821936, -78.612958), (35.811913, -78.603259)]

### I-40
name = "40"
points = [(36.060618, -79.595281), (36.062144, -79.582578), (36.062144, -79.572621), (36.061866, -79.562322), (36.062702, -79.541219), (36.064645, -79.528001), (36.067559, -79.500020), (36.069502, -79.483197), (36.064368, -79.465345), (36.065478, -79.442857), (36.063535, -79.427407), (36.060066, -79.400628), (36.064923, -79.382261), (36.068530, -79.374021), (36.068669, -79.352048), (36.066865, -79.328016), (36.065200, -79.317716), (36.068738, -79.303597), (36.067996, -79.286511), (36.068412, -79.279730), (36.076945, -79.255611), (36.079431, -79.222129)]

### I-85
# name = "85"
# points = [(35.776606, -80.279360), (35.788861, -80.227862), (35.803341, -80.189410), (35.826727, -80.168811), (35.832294, -80.145465), (35.850106, -80.120059), (35.856785, -80.089160), (35.861237, -80.076800), (35.859011, -80.056887), (35.873478, -80.009509), (35.887943, -79.988909), (35.894619, -79.974490), (35.900737, -79.955264), (35.914641, -79.949084)]

### Number of clusters
k = 4

### Results within n miles from the highway
n = 3

### Snap to Road -- more accurate version of highway
true_path, highway_lats, highway_longs = scrape_google_maps.lat_long_snapped_path(points)

### Scrape Google Maps for hotels near pathway
hotel_vectors_r3 = scrape_google_maps.collect_hotels_along_highway(true_path, filename="gmaps_hotels.txt", return_value=True, index_range=(1, 4), miles = n)
scaled_vectors_r3 = StandardScaler().fit(hotel_vectors_r3).transform(hotel_vectors_r3)
print("1")

### Create Cubic Spline of Highway
hw_longs_2d, hw_lats_2d = [], []
for long in highway_longs:
    hw_longs_2d.append([long])
for lat in highway_lats:
    hw_lats_2d.append([lat])
print("1.2")
hw_longs_scaled = StandardScaler().fit(hw_longs_2d).transform(hw_longs_2d)
print("1.7")
hw_lats_scaled = StandardScaler().fit(hw_lats_2d).transform(hw_lats_2d)
print("2")
hw_cubic_spline = CubicSpline(highway_longs, highway_lats)
hw_cubic_spline_scaled = CubicSpline(hw_longs_scaled.flatten(), hw_lats_scaled.flatten())

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

### Visualize both clusterings
r4_info = visualize_clusters.visualize_clusters(scaled_vectors_r4, kmeans_r4, k, title=f"{name} With Signed Distance", plot_rating=False)
# Plot highway
visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
plt.savefig(f"I-{name}_with_signed_distance")
plt.show(block=False)

r3_info = visualize_clusters.visualize_clusters(scaled_vectors_r3, kmeans_r3, k, title=f"{name} Without Signed Distance", plot_rating=False)
# Plot highway
visualize_clusters.plot_cubic_spline_highway(hw_cubic_spline_scaled)
plt.savefig(f"I-{name}_without_signed_distance")
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

visualize_clusters.increment_beta_values(scaled_vectors_r4, hw_cubic_spline_scaled, name, k, n, method="bool_value")

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
