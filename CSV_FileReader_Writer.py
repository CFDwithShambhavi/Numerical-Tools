# creator -----Shambhavi----

import csv
import numpy as np

def csv_fileWriter_with_X_data_name(path_name, file_name, delimiter, X_data_name, data):

    headers = []
    data_matrix = []

    data_matrix.append(data[X_data_name])
    headers.append(X_data_name)
    
    for i in data.keys():
        if i != X_data_name:
            headers.append(i)
            data_matrix.append(data[i])

    transpose_data = np.transpose(data_matrix)

    with open(path_name+file_name,'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter = delimiter)
        writer.writerow(headers)
        writer.writerows(transpose_data)
        
def csv_fileWriter(path_name, file_name, delimiter, data):

    headers = []
    data_matrix = []
    
    for i in data.keys():
        headers.append(i)
        data_matrix.append(data[i])

    transpose_data = np.transpose(data_matrix)

    with open(path_name+file_name,'w+') as csv_file:
        writer = csv.writer(csv_file, delimiter = delimiter)
        writer.writerow(headers)
        writer.writerows(transpose_data)
        

def csv_fileReader(path_name, file_name, delimiter):

    data_matrix = []
    data = {}
    
    csv_file = csv.reader(open(path_name + file_name), delimiter = delimiter)
    for row in csv_file:
        data_matrix.append(row)

    headers = data_matrix[0]
    data_without_headers = data_matrix[1:]
    transpose_data = np.transpose(data_without_headers)

    for i in range(0,len(headers)):
        data[headers[i]] = transpose_data[i]

    return (data, headers)
        


