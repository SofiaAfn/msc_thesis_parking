# Triangulation

main.py is the runner for triangulating detected vehicles and is used as follows:

`python3 main.py path_to_sequnce_f_file output_filename.csv`

> Note: As of right now the code also relies on a separate vehicle_detections.hdf5 file, let me know if we want to change this

## Files

- camera.py - Containes code related to camera projections
- compass_search_triangulation.py - Code that experiments with shifting a triangulated point using compass search in order to minimize image plane error
- line_intersect.py - Code that "intersects" lines in 3D
- main.py - Main runner
- system.py - Code for reading detections, images, and some experimentation
- vehicle_identification.py - Code that tries to re identificate objects over image sequences

# Status:

As of right now I feel like the performance of main.py is sub-par, I've tried
