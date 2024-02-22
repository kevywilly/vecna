from src.kinematics import forward, inverse, Vector3
import math


v0 = Vector3(0,0,0).numpy()
pose = forward(v0)
a0 = inverse(pose)
p0 = forward(a0)
print('angle_actual',v0,'\tpose_actual\t',pose, '\nangle_predicted\t', a0, '\tpose_predicted\t', p0)
print('-------------------------------------------------------------------------------------')


v0 = Vector3(0,40,42).numpy()
pose = forward(v0)
a0 = inverse(pose)
p0 = forward(a0)
print('angle_actual',v0,'\tpose_actual\t',pose, '\nangle_predicted\t', a0, '\tpose_predicted\t', p0)
print('-------------------------------------------------------------------------------------')

pose = Vector3(0,63,40).numpy()
a0 = inverse(pose)
p0 = forward(a0)
print('angle_predicted\t',a0, '\tpose_requested\t',pose,'\tpose_predicted\t', p0)
print('-------------------------------------------------------------------------------------')

pose = Vector3(0,67,39).numpy()
a0 = inverse(pose)
p0 = forward(a0)
print('angle_predicted\t',a0, '\tpose_requested\t',pose,'\tpose_predicted\t', p0)




"""
for c in range (-90,90,10):
    for f in range(-50,50,10):
        for t in range(-90,90,10):
            x,y,z = forward(Vector3(c,f,t)).array()
            print(f'{round(x,2)},{round(y,2)},{round(z,2)},{c},{f},{t}')
"""