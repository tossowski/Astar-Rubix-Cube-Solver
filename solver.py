import rubix as r
import time
import numpy as np


cube = r.Cube(3)
input = int(input("Number of scrambles: "))
myCube = np.array([np.array([[z for x in range(3)] for y in range(3)]) for z in range(6)])
myCube = cube.scramble(myCube, input)
sol = (cube.astar(myCube, cube.h))
cube.generateOBJ(sol[0])
start = time.time()
print("Found a solution: " + str(len(cube.astar(myCube, cube.h)) - 1) + " moves long")
print("Time taken: " + str(time.time() - start))
