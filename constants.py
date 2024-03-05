import numpy as np



with open("./data/dataset/sequences/00/calib.txt", 'r') as f:
    calib_data= f.readlines()
    P2_line = calib_data[2].strip().split(' ')
    P2_matrix = [float(value) for value in P2_line[1:]]  # Skip the first element (contains 'P2:')
    K = np.array(P2_matrix).reshape(3, 4)
    
