import csv
def ReadFromCSVFile(filename_with_path):
    #with open('D:\Karnik\Graduate Studies\Statistical Machine Learning\Project\data\A.csv', newline='') as csvfile:
    '''with open(filename_with_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            print(', '.join(row))
    return row'''

    with open(filename_with_path, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    return your_list




