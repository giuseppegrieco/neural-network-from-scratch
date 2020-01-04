import numpy as np
import sys
path = sys.argv[1];
print(path)

mat = np.load(path)
print("length: " + str(len(mat[0])))
print("min: " + str(min(mat[0])))
print("last: " + str(mat[0][-1]))
