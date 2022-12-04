import numpy as np

x = [475, 476, 477, 474, 469, 470, 472, 469, 471, 472, 474, 475, 476, 477, 470, 471]
x = sorted(list(set(x)))
print(np.average(x))