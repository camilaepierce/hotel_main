# hotel_main

## Project Description
The goal of this project was to investigate how incorporation of a relation to a geographical feature affects clustering through k++ unsupervised learning. This software was developed to add signed distance to highway from nearby hotels and run clusterings with various tunable parameters.

## Setting up Environment

Make sure you have the following libraries installed:
```
numpy
scipy
sklearn
matplotlib
shapely
math
googlemaps
```
Upgrade pip before installing packages:
```
python.exe -m pip install --upgrade pip
pip instal [packages]
```

Temorary Google Maps API is in GitHub but it needs to be changed for continued use. To aquire a new key, create a Google API account (Free for 90 days).<br>
API Key creation outlined here: https://developers.google.com/maps/get-started<br>
Be sure to add Google Places, Roads, Matrix Distance, and Service Usage APIs and to restrict key usage to just these APIs.<br>

## Running Project
Steps to create clustering with highway data:
1. Create .txt file with latitude longitude information in two columns, without commas.
      &ensp *Longitude must increase for Cubic Spline to be created (highway runs left to right)<br>
      &ensp *Placing a pin down in Google Maps yields a small text box with long/lat information<br>
2. Run scraping_script.py with python:
      &ensp *script takes two parameters: [input file] [highway name]<br>
      &ensp *if not in the same folder, input file must be the relative path to the intended file<br>
      &ensp *all related file creations will be implementing this name, example recommendation: I-440<br>
    ```
    python scraping_script.py ./test_input/test_scraping_input.txt test_input
    ```
      &ensp Script will print names of created files for later use.<br>
3. Run clustering_script.py with python:
      &ensp *Can either be run as a command line script with passed in arguments: [snapped path file] [hotel data file] [highway name]<br>
    ```
    python clustering_script.py ./snapped_highways/test_input_path.txt ./hotel/test_input_hotels.txt test_input
    ```
      &ensp *Or variables on lines 32-34can be editted and the file run<br>
      &ensp *Lines 36-47 contain tuning parameters for clustering, including the array of beta values to plot clusterings for<br>
      &ensp Script will create plots describing the clustering data.<br>
          &ensp *Plotting controls for with signed distance and without signed distance (no beta modification)<br>
          &ensp *Incrementing beta values with their adjusted_rand_score<br>
          &ensp *Plotting each beta in beta_array for comparison<br>
      &ensp Created plots will be created under the results folder unless otherwise specified.<br>
<br>
  &ensp Files such as test_input, I-85, I-40, and I-440 are included in the project as examples.
