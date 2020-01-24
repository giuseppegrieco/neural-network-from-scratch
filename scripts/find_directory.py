import os
import sys
import json
import xlsxwriter

file = sys.argv[1]

lr_to_find = float(sys.argv[2])
reg_to_find = float(sys.argv[3])
mom_to_find = float(sys.argv[4])
nodes_to_find = int(sys.argv[5])

for directory in os.listdir(file):
    if directory != '.DS_Store' and directory != "run.json" and directory != 'data.json' and directory != 'data copia.json':
        with open(file + directory + '/hyperparameters.json', 'r') as myfile:
            data = myfile.read()

        obj = json.loads(data)
        layers = int(obj['layers'][0].split(',')[0].split('=')[1])
        topology = layers
        lr = float(obj['learning_rate'])
        lambda_reg = float(obj['regularization'])
        momentum = float(obj['momentum'])

        if lr_to_find == lr and reg_to_find == lambda_reg and mom_to_find == momentum and nodes_to_find == topology:
            print(directory)


sys.exit()
