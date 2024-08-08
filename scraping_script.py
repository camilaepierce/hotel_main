"""
Run to scrape data from Google Maps.
"""
#!/usr/bin/env python3
import sys
from library import scrape_google_maps

def read_data(f_name):
    """ Reads txt file that contains highway data.
    Assumes in columns of Latitude, Longitude, returns as list of tuples (points).
    Expects no commas, values are separated by whitespace, and no column labels.

    Highway longitude must be in increasing order.
    """
    file = open(f_name, "r")
    file.readline()
    data = []
    for line in file.readlines():
        data.append(tuple(float(d) for d in line.split()))
    return data

def scrape_and_save_to_files(data, data_name, verbose=True):
    """
    Scrapes Google Maps and saves hotel data to a single file, and snapped
    path data to another.
    """
    hotel_file_name = f"hotel_data/{data_name}_hotels.txt"
    snapped_file_name = f"snapped_highways/{data_name}_path.txt"

    if verbose: print("Finding snapped path...")
    true_path, highway_lats, highway_lonngs = scrape_google_maps.lat_long_snapped_path(data)
    if verbose: print("Scraping hotel data...")
    scrape_google_maps.collect_hotels_along_highway(true_path, return_value=False,
                                                    filename=hotel_file_name)
    save_matrix_to_file(true_path, snapped_file_name)
    if verbose:
        print("Complete!")
        print()
        print("Associated files:")
        print(hotel_file_name)
        print(snapped_file_name)

def save_matrix_to_file(list_of_lists, new_path):
    """
    Saves list of points to a text file for later reading.
    """
    with open(new_path, "w", encoding="utf-8") as file:
        for row in list_of_lists:
            file.write(" ".join([str(c) for c in row]) + "\n")

if __name__ == "__main__":
    import os
    # sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    if len(sys.argv) >= 3:
        file_path = sys.argv[1]
        highway_data = read_data(file_path)
        file_name = sys.argv[2]# file_path.split("\\")[-1].split(".")[0]
        scrape_and_save_to_files(highway_data, file_name)

    else:
        print("Error: Too few arguments")
