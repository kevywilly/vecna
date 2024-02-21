import numpy as np



class Vecna:

    class dims:
        coxa = 25
        femur = 40
        tibia = 85
        tibia_z_offset = 85-30
        
        angle_0 = 0
        angle1 = 5
        angle2 = -90

    joints = np.array(
        [
            [0,1,2],
            [4,5,6],
            [8,9,10],
            [12,13,14]
        ]
    )
    dir = np.array(
        [
            [1,1,1],
            [1,-1,-1],
            [1,1,1],
            [1,-1,-1]
        ]
    )

    adj = np.array(
        [
            [0,0,5],
            [0,0,-10],
            [0,0,10],
            [0,0,-5]
        ]
    )
    home = np.zeros((4,3)) #+ np.array([0,0,0])
    squat = home + np.array([0,35,45])
    stretch = home + np.array([0,-45,-35])
    twist = home + np.array([55,0,0])
    step = home + np.array([25,60,30])

