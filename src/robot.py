import numpy as np

class Robot:
    joints = np.array([[0,1,2],[4,5,6],[8,9,10],[12,13,14]])
    dir = np.array([[1,-1,-1],[1,1,1],[1,-1,-1],[1,1,1]])
    adj = np.array(
        [
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
    )
    home = np.zeros((4,3))
    squat = home + np.array([0,50,50])
    stretch = home + np.array([0,30,30])
    twist = home + np.array([25,0,0])
    step = home + np.array([25,60,30])

robot = Robot()