import os
import sys
import json
import xlsxwriter
import numpy as np

file = sys.argv[1]

workbook = xlsxwriter.Workbook('grid_search.xlsx')
worksheet_mean = workbook.add_worksheet(
    "mean"
)
worksheet_variance = workbook.add_worksheet(
    "variance"
)
worksheet_nodes = workbook.add_worksheet(
    "nodes"
)
heads = workbook.add_format({
    'bold': True,
    'align': 'center',
    'valign': 'center'
})
top_left = workbook.add_format({
    'bg_color': "#000000"
})
h_error = workbook.add_format({
    'bg_color': "#e74c3c",
    'font_color': 'white'
})
normal = workbook.add_format({

})

for worksheet in workbook.worksheets():
    worksheet.freeze_panes(0, 1)
    worksheet.freeze_panes(0, 2)
    worksheet.freeze_panes(1, 2)
    worksheet.freeze_panes(2, 2)

    worksheet.write(0, 0, "", top_left)
    worksheet.write(0, 1, "", top_left)
    worksheet.write(1, 0, "", top_left)
    worksheet.write(1, 1, "", top_left)

    columns = [
        0.9, 0.6, 0.3, 0.1, 0
    ]
    rows = [
        0.1,0.01, 0.001, 0.0001, 0.00001, 0.000001
    ]
    sub_columns = [
        0.1,0.01, 0.001, 0.0001, 0.00001, 0.000001
    ]
    sub_rows = [
        0.5, 0.1, 0.01, 0.001, 0.0001
    ]

    i = 0
    for column in columns:
        worksheet.merge_range(
            0,
            2 + len(sub_columns) * i,
            0,
            (len(sub_columns) * (i + 1)) + 1,
            column,
            heads
        )
        j = 0
        for sub_column in sub_columns:
            worksheet.write_number(
                1,
                (j + 2) + (len(sub_columns) * i),
                sub_column,
                heads
            )
            j += 1
        i += 1

    i = 0
    for row in rows:
        worksheet.merge_range(
            2 + len(sub_rows) * i,
            0,
            (len(sub_rows) * (i + 1)) + 1,
            0,
            row,
            heads
        )
        j = 0
        for sub_row in sub_rows:
            worksheet.write_number(
                (j + 2) + (len(sub_rows) * i),
                1,
                sub_row,
                heads
            )
            j += 1
        i += 1

for directory in os.listdir(file):
    if directory != '.DS_Store' and directory != "run.json" and directory != 'data.json' and directory != 'data copia.json':
        with open(file + directory + '/hyperparameters.json', 'r') as myfile:
            data = myfile.read()
        avg = 0
        for i in range(1,6):
            mse = np.load(file + directory + '/fold-' + str(i)+'/validation_errors_mse.npy')
            avg += len(mse)
        avg = avg / 5
        obj = json.loads(data)
        reg_pseudo = float(obj['regularization_pseudo_inverse'])
        lr = float(obj['learning_rate'])
        reg_correlation = float(obj['regularization_correlation'])
        momentum = float(obj['momentum'])

        with open(file + directory + '/result.json', 'r') as myfile:
            data = myfile.read()
        obj = json.loads(data)
        average = float(obj['mean'])
        variance = float(obj['variance'])

        row = 2 + (rows.index(reg_pseudo) * len(sub_rows)) + sub_rows.index(lr)
        column = 2 + (columns.index(momentum) * len(sub_columns)) + sub_columns.index(reg_correlation)
        worksheet_mean.write_number(
            row,
            column,
            average,
            normal if average < 1.4 else h_error,
        )
        worksheet_variance.write_number(
            row,
            column,
            variance
        )
        worksheet_nodes.write_number(
            row,
            column,
            avg
        )

workbook.close()
sys.exit()
