import os
import sys
import json
import xlsxwriter

file = sys.argv[1]

lr_to_find = float(sys.argv[2])
reg_correlation_to_find = float(sys.argv[3])
reg_pseudo_to_find = float(sys.argv[4])
mom_to_find = float(sys.argv[5])


for directory in os.listdir(file):
    if directory != '.DS_Store' and directory != "run.json" and directory != 'data.json' and directory != 'data copia.json':
        with open(file + directory + '/hyperparameters.json', 'r') as myfile:
            data = myfile.read()

        obj = json.loads(data)
        reg_pseudo = float(obj['regularization_pseudo_inverse'])
        lr = float(obj['learning_rate'])
        reg_correlation = float(obj['regularization_correlation'])
        momentum = float(obj['momentum'])

        if lr_to_find == lr and reg_pseudo_to_find == reg_pseudo and mom_to_find == momentum and reg_correlation_to_find == reg_correlation:
            print(directory)


sys.exit()
