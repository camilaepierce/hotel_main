"""
Created by Camila Pierce
Last Updated 8.5.2024

Google Maps Python Services Library:
https://googlemaps.github.io/google-maps-services-python/docs/index.html
googlemaps GitHub Python Client:
https://github.com/googlemaps/google-maps-services-python

run 'pip install -U googlemaps' prior to running associated scripts

***API Key*** First created July 24th, 2024, will run out October 22, 2024
AIzaSyCCC-jl0fP7Hyp4jZCdRtcCzWHHq7cDT7k
To create your own key: Must enable Google Maps API in google developer console.

Reformatted using black.
"""

import googlemaps
import numpy as np

# possibly filter by number of ratings? holdback data
# with and without signed distance clustering
gmap = googlemaps.Client(key="AIzaSyCCC-jl0fP7Hyp4jZCdRtcCzWHHq7cDT7k")


def collect_hotels_along_highway(
    path,
    r=50,
    filename="gmaps_hotels.txt",
    return_value=False,
    abbreviate=False,
    index_range=(1, 5),
    miles=3,
):
    """Reads txt file that contains hotel data.
    Assumes in columns of Name, X location, Y location, and Rating."""
    find_road = gmap.snap_to_roads(path, interpolate=True)
    data = set()

    for hw_point in find_road:
        latlong = hw_point["location"]
        result = gmap.places(query="hotels", radius=r, location=latlong)
        # print(result)
        for place in result[
            "results"
        ]:  ##place is a dictionary with several keys--name, rating, geometry:location:lat/long
            if within_distance(latlong, place["geometry"]["location"], miles):
                data.add(
                    (
                        place["name"],
                        float(place["geometry"]["location"]["lng"]),
                        float(place["geometry"]["location"]["lat"]),
                        float(place["rating"]),
                        int(place["user_ratings_total"]),
                    )
                )
            else:
                print(f"Excluding {place["name"]} from the results.")
                # pass

    with open(filename, "w", encoding="utf-8") as file:
        file.write("Abbrv.\tLat\tLong\tRating\tNumber_Of_Reviews\n")
        for hotel in data:
            if abbreviate:
                file.write(
                    f"{''.join(character for character in hotel[0] if character.isupper())}\
                    \t{hotel[1]}\t{hotel[2]}\t{hotel[3]}\t{hotel[4]}\n"
                )
            else:
                file.write(f"{hotel[0].replace(" ", "_")}\t{hotel[1]}\t{hotel[2]}\t{hotel[3]}\
                        \t{hotel[4]}\n")
    if return_value:
        np.set_printoptions(suppress=True)
        return np.asarray(list(data))[:, index_range[0] : index_range[1]].astype(float)
    return None


def within_distance(loc_a, loc_b, mi):
    """
    Returns True/False whether distance between two points is less than maximum distance.

    Parameters:
    * loc_a (list) : coordinate points, latitude and longitude
    * loc_b (list) : coordinate points, latitude and longitude
    * mi (float) : number of miles from highway to restrict search

    Returns:
        Bool value of if hotel is within specificed distance from highway.
    """
    meters = mi * 1609
    distance = gmap.distance_matrix(loc_a, loc_b, mode="driving", units="imperial")
    return distance["rows"][0]["elements"][0]["distance"]["value"] <= meters


def lat_long_snapped_path(path):
    """
    Returns snapped path of highway from manually chosen points. (More accurate tracing)

    Parameters:
    * path (list) : list of coordinate points manually chosen along highway

    Returns:
        Tuple (out, lat, long):
        * out (list) : combined list of (lat, long) per point
        * lat (list) : just the latitude elements of each point
        * long (list) : just the longitude elements of each point
    """
    snapped_points = gmap.snap_to_roads(path, interpolate=True)
    out, lat, long = [], [], []
    for hw_point in snapped_points:
        out.append(
            (hw_point["location"]["latitude"], hw_point["location"]["longitude"])
        )
        lat.append(hw_point["location"]["latitude"])
        long.append(hw_point["location"]["longitude"])
    return out, lat, long


# points = [(35.828147, -78.682148), (35.835497, -78.668780), (35.836619, -78.656646),\
# (35.835227, -78.642913), (35.832792, -78.630725), (35.821936, -78.612958), (35.811913,\
# -78.603259)]
# test = collect_hotels_along_highway(points, abbreviate=True, return_value=True)

# # print(test)
# # print(type(test))
# # print(type(test[0][-1]))

# out, lat, long = lat_long_snapped_path(points)
# print(long)
# for i in range(1, len(long)):
#     print(long[i-1], long[i], long[i-1] - long[i])
