import numpy as np



class Vecna:

    class training:
        means = np.array([-1.14875857, 36.55266804, -2.22222222])
        stds = np.array([48.79383234, 61.68878012, 28.19638266])
        maxs = np.array([126.1,  148.71,  39.39])

    class dims:
        coxa = 25
        femur = 40
        tibia = 85
        tibia_z_offset = 85-30
        thetas = np.array([0,0,90])
        lengths = np.array([25,40,85])

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

