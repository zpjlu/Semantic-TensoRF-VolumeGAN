from volumegan import PointsSampling
import math
PI = math.pi
import torch
args = dict(num_steps=12,
                             ray_start=0.88,
                             ray_end=1.12,
                             radius=1,
                             horizontal_mean=PI/2,
                             horizontal_stddev=0.3,
                             vertical_mean=PI/2,
                             vertical_stddev=0.15,
                             camera_dist='gaussian',
                             fov=12,  # TODO:需要根据数据集调整这一系列参数
                             perturb_mode=None)
func = PointsSampling(**args)
ret = []
for i in range(3000):
    ps = func(4,64)
    ret.append(ps['pts'])
pts = torch.cat(ret,-2)
print(torch.amax(pts[...,0]))
print(torch.amax(pts[...,1]))
print(torch.amax(pts[...,2]))
print(torch.amin(pts[...,0]))
print(torch.amin(pts[...,1]))
print(torch.amin(pts[...,2]))

