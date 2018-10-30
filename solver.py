import rubix as r
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def main():
    cube = r.Cube(3)
    myCube = np.array([np.array([[z for x in range(3)] for y in range(3)]) for z in range(6)])
    #cube.generateOBJ(myCube)
    #print(cube.h(myCube))
    #test = np.array([[0,0,1],[0,0,0],[0,0,0]])
    #print(cube.hf(test))
    myCube = cube.scramble(myCube, 100)
    cube.generateOBJ(myCube)
    #cube.astar(cube.h, myCube)


if __name__ == "__main__":
    main()
    # total = 0
    # for i in range(10):
    #     start_time = time.time()
    #     main()
    #     print("Solved in: " + str(time.time() - start_time))
    #     total += time.time() - start_time
    # print("Average: " + str(float(total)/10))
