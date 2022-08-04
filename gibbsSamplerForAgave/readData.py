import csv
import numpy

with open("C:/Users/User/Downloads/data_to_share.txt") as inp:
    data = [x.strip().split('\t') for x in inp]


filteredData = numpy.delete(data, '')
