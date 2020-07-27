import numpy as np
from queue import PriorityQueue
import math
import random
import enum

# Enum representing colors of cube for rendering
class Color(enum.Enum):
    Green = 0
    Red = 1
    Yellow = 2
    Orange = 3
    White = 4
    Blue = 5

# Enum representing the 6 faces of a rubix cube
class Face(enum.Enum):
    F = 0 # Front faces
    R = 1 # Right face
    B = 2 # Back face
    L = 3 # Left face
    T = 4 # Top face
    D = 5 # Bottom face

class Axis(enum.Enum):
    Horizontal = 0 # Turn along horizontal AXIS
    Vertical = 1 # Turn along vertical AXIS
    Depth = 2 # Turn along depth axis (like turning front face)


class Rotation(enum.Enum):
    Clockwise = 1 # Clockwise turn
    Counterclockwise = 0 # Counterclockwise turn


# Returns the string representation of a list without brackets
def listToString(alod):
    return str(alod).strip("[]")



class Cube:
    def __init__(self, n):
        self.dim = n
        # Faces of the cube
        self.F = np.array([np.array([[z for x in range(self.dim)] for y in range(self.dim)]) for z in range(6)])

    # Returns True if the face contains dim x dim copies of the same color
    def isUniformInColor(self, arr):
        color = arr[0][0]
        for i in range(self.dim):
            for j in range(self.dim):
                if arr[i][j] != color:
                    return False
        return True

    # Given a set of points representing a face, write the points to the obj
    # file along with the appropriate colors
    # points: list of lists of length 3, representing xyz points
    # count: Current number of vertices. Need to keep track to generate faces
    # correctly
    # Returns the updated count variable
    def writePointsToFile(self, points, count, f, face):
        c = self.dim + 1
        for i in range(self.dim):
            for j in range(self.dim):
                f.write("v " + listToString(points[i*c + j]) + "\n")
                f.write("v " + listToString(points[i*c + j + 1]) + "\n")
                f.write("v " + listToString(points[(i+1) * c + j + 1]) + "\n")
                f.write("v " + listToString(points[(i+1) * c + j]) + "\n")
                f.write("usemtl " + Color(face[i][j]).name + "\n")
                f.write("f " + listToString(np.array([count,count+1,count+2,count+3])) + "\n")
                count += 4
        return count



    # Generates the obj file representing the cube. File written is named cube.obj
    def generateOBJ(self,s, num):
        count = 1
        with open("cube" + str(num) + ".obj", "w") as f:
            f.write("mtllib colors.mtl \n")
            points = np.array([(y,-1.5,x) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[0])
            points = np.array([(y,1.5,x) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[1])
            points = np.array([(-1.5,y,x) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[2])
            points = np.array([(1.5,y,x) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[3])
            points = np.array([(y,x,-1.5) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[4])
            points = np.array([(y,x,1.5) for x in np.linspace(1.5,-1.5,self.dim+1) for y in np.linspace(-1.5,1.5,self.dim+1)])
            count = self.writePointsToFile(points, count, f, s[5])


    # Given a cube, scramble it
    # Input: n: Number of moves to scramble it
    def scramble(self, s, n):
        if n == 0:
            self.F = s
            return s

        next_state = [x.copy() for x in self.get_successors(s)]
        return self.scramble(next_state[random.randint(0, len(next_state)-1)], n-1)



    # Returns true if the cube is solved
    def is_goal_state(self, s):
        for face in s:
            if (not self.isUniformInColor(face)):
                return False
        return True

    # Prints out an array representing a face of the cube
    def printFace(self, face):
        print(self.F[face])

    # Swaps corresponding elements of a list
    def swapEntries(self,a,b):
        for i in range(len(a)):
            a[i],b[i] = b[i],a[i]

    # Iterator that when iterated over, gives a list of all possible get_successors
    # states (ones that result after a quarter turn of the current cube)
    def get_successors(self, s):
        state = np.copy(s) # save s to not modify original
        for i in range(self.dim):
            self.move(Axis.Horizontal, Rotation.Clockwise, i, state)
            yield state
            self.move(Axis.Horizontal, Rotation.Counterclockwise, i,state)
            self.move(Axis.Horizontal, Rotation.Counterclockwise, i,state)
            yield state
            self.move(Axis.Horizontal, Rotation.Clockwise, i,state)
        for i in range(self.dim):
            self.move(Axis.Vertical, Rotation.Clockwise, i,state)
            yield state
            self.move(Axis.Vertical, Rotation.Counterclockwise, i,state)
            self.move(Axis.Vertical, Rotation.Counterclockwise, i,state)
            yield state
            self.move(Axis.Vertical, Rotation.Clockwise, i,state)
        for i in range(self.dim):
            self.move(Axis.Depth, Rotation.Clockwise, i,state)
            yield state
            self.move(Axis.Depth, Rotation.Counterclockwise, i,state)
            self.move(Axis.Depth, Rotation.Counterclockwise, i,state)
            yield state
            self.move(Axis.Depth, Rotation.Clockwise, i,state)



    # Performs a move operation on a cube. Takes in 4 parameters:
    # Axis: Either Axis.Horizontal or Axis.Vertical
    # Rotation: Either Rotation.Clockwise or Rotation.Counterclockwise
    # Strip: The index representing either the row or col number to swapself.
    # s: A state to perform the move on
    # Counts starting from 0 at the top edge for rotations along vertical axis.
    # Counts starting from 0 at the left edge for rotations along horizontal
    # axis.
    def move(self,axis,rotation,strip, s):
        # Horizontal turn
        if (axis.value == 1):
            # Clockwise turn
            if (rotation.value == 1):
                self.swapEntries(s[Face.F.value][strip,:], s[Face.L.value][strip,:])
                self.swapEntries(s[Face.L.value][strip,:], s[Face.B.value][strip,:])
                self.swapEntries(s[Face.B.value][strip,:], s[Face.R.value][strip,:])
            # Counterclockwise turn
            else:
                self.swapEntries(s[Face.F.value][strip,:], s[Face.R.value][strip,:])
                self.swapEntries(s[Face.R.value][strip,:], s[Face.B.value][strip,:])
                self.swapEntries(s[Face.B.value][strip,:], s[Face.L.value][strip,:])
        # Vertical turn
        elif (axis.value == 0):
            # Clockwise turn
            if (rotation.value == 1):
                self.swapEntries(s[Face.F.value][:,strip], s[Face.T.value][:,strip])
                self.swapEntries(s[Face.T.value][:,strip], s[Face.B.value][:,strip])
                self.swapEntries(s[Face.B.value][:,strip], s[Face.D.value][:,strip])
            # Counterclockwise turn
            else:
                self.swapEntries(s[Face.F.value][:,strip], s[Face.D.value][:,strip])
                self.swapEntries(s[Face.D.value][:,strip], s[Face.B.value][:,strip])
                self.swapEntries(s[Face.B.value][:,strip], s[Face.T.value][:,strip])
        else:
            # Clockwise turn
            if (rotation.value == 1):
                self.swapEntries(s[Face.D.value][:,strip], s[Face.R.value][:,strip])
                self.swapEntries(s[Face.R.value][:,strip], s[Face.T.value][:,strip])
                self.swapEntries(s[Face.T.value][:,strip], s[Face.L.value][:,strip])
            # Counterclockwise turn
            else:
                self.swapEntries(s[Face.D.value][:,strip], s[Face.L.value][:,strip])
                self.swapEntries(s[Face.L.value][:,strip], s[Face.T.value][:,strip])
                self.swapEntries(s[Face.T.value][:,strip], s[Face.R.value][:,strip])


    # Heuristic function for a single face: take 9 - (frequency of color that
    # appears most on face)
    def hf(self, s):
        index = int(self.dim/2)
        faceColor = s[index][index]
        #print(np.bincount(s.flatten()))
        mostfrequent = max(np.bincount(s.flatten()))
        return 9 - mostfrequent

    # Heuristic function for the cube:
    def h(self, state):
        total = 0
        for i in range(6):
            total += self.hf(state[i])
        return total / 12

    # Returns a set of moves required to solve the cube
    def astar(self,s, heur):
        node = (s.tostring(), 0)
        visited = set()
        frontier = PriorityQueue()
        frontier.put((0, node))

        parents = {node[0]:None}
        costs = {node[0]: 0}
        while (not self.is_goal_state(np.frombuffer(node[0], int).reshape(6,self.dim,self.dim))):
            node = frontier.get()[1]
            if(node[0] in visited):
                continue
            else:
                visited.add(node[0])
                for state in self.get_successors(np.frombuffer(node[0], int).reshape(6,self.dim,self.dim)):
                    st = state.tostring()
                    if (st not in visited):
                        if st not in costs or node[1] + 1 < costs[st]:
                            parents[st] = node[0]
                            costs[st] = node[1] + 1
                        frontier.put((heur (state) + node[1] + 1, (st, node[1] + 1)))


        node = node[0]
        return self.makePath(parents, node)

    def makePath(self,parents, end):
        """
        Given a dictionary of parents, and an ending key, returns the complete
        path fom the start until the key.

        Input:
             parents: a dictionary keeping track of nodes and their goal_parents
             end: The end of the path, represented as a string in binary
        """
        path = [np.frombuffer(end, int).reshape(6,self.dim,self.dim)]
        while (parents[end] != None):
            path.insert(0, np.frombuffer(parents[end], int).reshape(6,self.dim,self.dim))
            end = parents[end]
        return path
