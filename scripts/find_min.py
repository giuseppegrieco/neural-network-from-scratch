import os
import sys
import json

file = sys.argv[1]

list = os.listdir(file)
min = sys.maxsize
minDirectoryName = None
all = None
for directory in list:
    if directory != '.DS_Store':
        with open(file + directory + '/data.json', 'r') as myfile:
            data = myfile.read()

        # parse file
        obj = json.loads(data)
        average = float(obj['average'])
        if average < min:
            min = average
            minDirectoryName = directory
            all = obj

print("min: " + str(min))
print("directory name: " + minDirectoryName)
print(obj)
