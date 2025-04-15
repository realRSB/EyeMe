import csv

def log_data(filename, data):
    """
    Appends a row of data (a dictionary) to a CSV file.
    If the file doesn't exist, writes a header first.
    
    Parameters:
      filename (str): The CSV file to write to.
      data (dict): A dictionary of data values. Keys will be used as column headers.
    """
    write_header = False
    try:
        with open(filename, 'r'):
            pass
    except FileNotFoundError:
        write_header = True

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)
