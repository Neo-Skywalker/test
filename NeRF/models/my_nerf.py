import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import sys
import json
from models.settings import *
Num = MY_RS
class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf

    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))

class MyNeRF():
    def __init__(self):
        super(MyNeRF, self).__init__()
        self.volumes_sigma = torch.zeros((Num,Num,Num,1))
        self.volumes_color = torch.zeros((Num,Num,Num,3))
        self.z_sampling_num = min(Num,128) # 如果Num小于128，就没必要优化了
        self.fine_sigma_coor = torch.zeros((Num,Num,self.z_sampling_num,3))
        self.fine_color_coor = torch.zeros((Num,Num,self.z_sampling_num,1))
    def save(self, pts_xyz, sigma, color):
        # Num, _ = pts_xyz.shape
        # print(f'\n ptxyz shape{pts_xyz.shape}')
        # Num = int(Num ** (1/3) + 1)
        # 现在只是在用体素重建整个模型。如果重建模型的时候很粗糙，采样无论多细致，也无济于事
        self.pts_xyz = pts_xyz.reshape((Num,Num,Num,3)) #前三位为数组坐标，最后一个3为绝对坐标
        self.volumes_sigma = sigma.reshape((Num,Num,Num,1))
        self.volumes_color = color.reshape((Num,Num,Num,3))
        # for i in range(Num): # 存下了物体表面附近的高精度体素
        #     for j in range(Num):
        #         sampling_target = self.volumes_sigma[i,j,:,0]
        #         # 先整理出最大的128个sigma对应的数组坐标，存取这些z坐标、sigma值
        #         val, idx = torch.sort(sampling_target, descending=True) #descending为alse，升序，为True，降序
        #         self.fine_sigma[i,j,:,0] = val[0:128]
        #         self.fine_sigma_coor[i,j,:] = idx[0:128]
        #         for k in range(min(Num,128)):
        #             self.fine_coor_sigma_color[f'{i} {j} {idx[k]}'] = (np.float64(val[k]),self.volumes_color[i,j,idx[k]])
        
        self.sigma_sort_z_coor = torch.argsort(self.volumes_sigma,dim=-2,descending=True) # sort along Z-axis, shape:(Num,Num,Num,1)
        self.sigma_top128_z_coor = self.sigma_sort_z_coor[:,:,0:self.z_sampling_num,:] # shape:(Num,Num,self.z_sampling_num,1)
        # meshgridlize
        dim1 = torch.arange(Num)
        dim2 = torch.arange(Num)
        dim3 = torch.arange(self.z_sampling_num)
        dim4 = torch.arange(1)
        mesh = torch.meshgrid(dim1,dim2,dim3,dim4)
        my_dim4 = torch.arange(1)
        self.sigma_top128_xyz_coor = self.volumes_sigma[mesh[0],mesh[1],self.sigma_top128_z_coor,my_dim4]
        print(f'No error, top128 shape:{self.sigma_top128_xyz_coor.shape}')
        sys.exit()
        #然后把其他地方的高精度体素数据转化为低精度：
        if Num // 128 > 1:
            MAG = Num // 128
            self.volumes_sigma = self.volumes_sigma[0::MAG,0::MAG,0::MAG,:]
            self.volumes_color = self.volumes_color[0::MAG,0::MAG,0::MAG,:]
        # self.my_sigma = sigma.reshape(Num ** 3)
        # with open('my_sigma.json','w') as f:
        #     json.dump({'sigma':self.volumes_sigma[:,:,:,0].tolist(),'color':self.volumes_color.tolist()},f)
        # sys.exit()
    def query(self, pts_xyz):
        pass
        # print(f'Num:{Num}')
        #TODO：在my_renderer.py中被调用
        ###以下为未优化的代码
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)
        self.volumes_sigma = self.volumes_sigma.to(pts_xyz.device)
        self.volumes_color = self.volumes_color.to(pts_xyz.device)
        X_index = ((pts_xyz[:, 0] + 0.125) * 4 * Num).clamp(0, Num-1).long() # 判断属于哪个体素（由于save是均匀划分，所以每个坐标就代表了那个体素中心）
        Y_index = ((pts_xyz[:, 1] - 0.75) * 4 * Num).clamp(0, Num-1).long()
        Z_index = ((pts_xyz[:, 2] + 0.125) * 4 * Num).clamp(0, Num-1).long()
        # print(X_index.device, self.volumes_sigma.device)
        sigma[:, 0] = self.volumes_sigma[X_index, Y_index, Z_index].reshape(N)
        color[:, :] = self.volumes_color[X_index, Y_index, Z_index].reshape(N,3)
        return sigma, color

        ###先在高精度体素查找
        # N, _ = pts_xyz.shape
        # sigma = torch.zeros(N, 1, device=pts_xyz.device)
        # color = torch.zeros(N, 3, device=pts_xyz.device)
        # self.volumes_sigma = self.volumes_sigma.to(pts_xyz.device)
        # self.volumes_color = self.volumes_color.to(pts_xyz.device)
        # X_index_fine = ((pts_xyz[:, 0] + 0.125) * 4 * Num).clamp(0, Num-1).long() # 高精度体素的绝对坐标->数组坐标
        # Y_index_fine = ((pts_xyz[:, 1] - 0.75) * 4 * Num).clamp(0, Num-1).long()
        # Z_index_fine = ((pts_xyz[:, 2] + 0.125) * 4 * Num).clamp(0, Num-1).long()
        # X_index_course = ((pts_xyz[:, 0] + 0.125) * 4 * 128).clamp(0, 128-1).long() # 高精度体素的绝对坐标->数组坐标
        # Y_index_course = ((pts_xyz[:, 1] - 0.75) * 4 * 128).clamp(0, 128-1).long()
        # Z_index_course = ((pts_xyz[:, 2] + 0.125) * 4 * 128).clamp(0, 128-1).long()
        # # print(X_index.device, self.volumes_sigma.device)
        # # sigma[:, 0] = self.volumes_sigma[X_index, Y_index, Z_index].reshape(N)
        # # color[:, :] = self.volumes_color[X_index, Y_index, Z_index].reshape(N,3)
        # print(f'N = {N}')
        
        # for i in range(N):
        #     xyz_str = f'{X_index_fine[i]} {Y_index_fine[i]} {Z_index_fine[i]}'
        #     if xyz_str in self.fine_coor_sigma_color:
        #         sigma[i,0] = self.fine_coor_sigma_color[xyz_str][0]
        #         color[i,:] = self.fine_coor_sigma_color[xyz_str][1]
        #     else:
        #         sigma[i,0] = self.volumes_sigma[X_index_course[i], Y_index_course[i], Z_index_course[i]]
        #         color[i,:] = self.volumes_color[X_index_course[i], Y_index_course[i], Z_index_course[i]]
        #     print(f'{i} finished')
        # return sigma, color


