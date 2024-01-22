import os
import csv
from datetime import datetime

# The directory where the files are located
root_directory = "/home/sofia_afn/Documents/thesis_data_labelled/Kristianstad_cam"

file_name_without_extension = os.path.splitext(root_directory)[0]

# For each directory in the directory tree
for directory, subdirectories, files in os.walk(root_directory):
    try:
        # The CSV file where the file names and timestamps will be stored
        csv_file = os.path.join(directory, f"{file_name_without_extension}_data.csv")

        # Read the existing file details from the CSV file
        existing_files = {}
        if os.path.exists(csv_file):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip the header
                existing_files = {rows[0]: rows[1] for rows in reader}

        # Open the CSV file in append mode
        with open(csv_file, "a") as f:
            writer = csv.writer(f)

            # If the CSV file was empty, write the header
            if not existing_files:
                writer.writerow(["File Name", "Timestamp"])

            # For each file in the current directory
            for file in files:
                # Get the file name and timestamp
                full_path = os.path.join(directory, file)
                timestamp = datetime.fromtimestamp(
                    os.path.getmtime(full_path)
                ).strftime("%Y-%m-%d %H:%M:%S")

                # If the file details don't already exist in the CSV file, append them
                if (
                    full_path not in existing_files
                    or existing_files[full_path] != timestamp
                ):
                    writer.writerow([full_path, timestamp])
    except:
        print(f"Error processing")
