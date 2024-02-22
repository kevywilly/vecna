from src.kinematics_lite import forward, inverse
from src.vectors import Vector3
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
