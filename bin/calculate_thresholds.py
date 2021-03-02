#! /usr/bin/env python
"""
calculate_thresholds.py
"""

"""
For each field in the galactic plane:

- Select 150 lightcurves from each of 64 RCID for each filter (9,600 lightcurves for each filter)
    - Load RCID corners
    - Query "sources" database for sources
    - Grab first 150 lightcurves with:
        - At least 20 unique days observed in the filter
- Inject microlensing signals into a copy of all of the lightcurves
    - Find the nearest PopSyCLE simulation for population sourcing
    - Add microlensing from models.py with observable conditions
- Calculate thresholds
    - Calculate eta, chi-squared reduced and J on all lightcurves
    - For a FPR for each statistics ranging from 10% to 0.01%:
        - Determine the resulting False Negative rate for microlensing events
    - Save data to disk
"""


from puzle.utils import gather_PopSyCLE_lb, find_nearest_lightcurve_file
from puzle.sample import generate_random_lightcurves_lb, fetch_sample_objects

lb_arr = gather_PopSyCLE_lb()
for i, (l, b) in enumerate(lb_arr):
    print('Processing (l, b) = (%.2f, %.2f) |  %i / %i' % (l, b, i, len(lb_arr)))
    lightcurve_file = find_nearest_lightcurve_file(l, b)
    objs = fetch_sample_objects(lightcurve_file)
    lightcurves_regular, lightcurves_ulens = generate_random_lightcurves_lb(l, b, objs)
