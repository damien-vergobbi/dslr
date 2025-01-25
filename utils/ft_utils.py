import sys
import csv

def read_csv(filename):
    """ Read a CSV file and return a list of dictionaries """
    try:
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist")
        sys.exit(1)

def is_numeric(value):
    """ Check if a value is numeric """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def get_numeric_columns(data, ignore_columns = []):
    """Get numeric columns from data"""
    numeric_data = {}
    
    if not data:
        return numeric_data
    
    # Read each column in the data
    for column in data[0].keys():
        # Ignore columns that are in the ignore_columns list
        if column in ignore_columns:
            continue
            
        values = []
        for row in data:
            if row[column] and is_numeric(row[column]):
                values.append(float(row[column]))
        if values:  # If there are numeric values
            numeric_data[column] = values
            
    return numeric_data

