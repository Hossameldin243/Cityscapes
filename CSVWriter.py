import os
import csv

# Function to save CSV File
def save_csv(file_name, data):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'annotation_path'])
        writer.writerows(data)
