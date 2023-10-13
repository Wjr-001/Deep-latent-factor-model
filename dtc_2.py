import numpy as np
import torch
from torch import nn
import pandas as pd
from torch import tensor
def net(args):
    #agrs为列表
    U_link = None
    for U in args:
        if U_link == None:
            U_link = U
        else:
            U_link = torch.mm(U_link,U)
    return U_link

def agd_U(k,T_0,args):
    #系数问题要单独处理,k为对应第几个U，T_0为预先求矩阵，args为列表。
    assert 1<=k<=len(args)-1 
    if k==1:
        net1=net(args[k:-1])#B计算之后的
        net2=net([args[0]]+[net1]+[args[-1]])
        return net([T_0]+[net2]+[net1.T])
    elif k == len(args)-1:
        net1=net(args[:-2])
        net2=net([net1]+[args[-2]]+[args[-1]])
        return net([net1.T]+[T_0]+[net2])
    else:
        net1=net(args[:k-1])
        net2=net(args[k:-1])
        net3=net([net1]+[args[k-1]]+[net2]+[args[-1]])
        return net([net1.T]+[T_0]+[net3]+[net2.T])    
    
def agd_B(T_0,args):
    #args为列表只有U
    net1 = net(args)
    return net([net1.T]+[T_0]+[net1])   

def relu(X,h,lr,mu,mask,X_agd):
    h = mu*h+(1-mu)*torch.mul(X_agd,X_agd) 
    X_agd[mask] = 0
    X-=lr*X_agd/(np.sqrt(h)+1e-6)   
    return h,X



def function(R_1,R_true,b,c,lr,step):
    t=0
    mu=0.9
    R_1 = np.array(R_1)
    a = R_1.shape[0]
    R = R_1.copy()
    number = np.count_nonzero(~np.isnan(R_1))
    Y = np.ones_like(R)
    Y[np.isnan(R)] =0
    R[np.isnan(R)] = 0
    U_1_origin = torch.nn.init.normal_(torch.empty(a,b),mean=0, std=0.1)
    U_2_origin = torch.nn.init.normal_(torch.empty(b,c),mean=0, std=0.1)
    # U_3_origin = torch.nn.init.kaiming_uniform_(torch.empty(c,d))
    B_origin = torch.nn.init.normal_(torch.empty(c,c),mean=0, std=0.1)
    # B_origin = torch.nn.init.uniform_(torch.empty(c,c),0,2*np.sqrt(3))
    
    # U_1 = torch.nn.init.uniform_(torch.empty(a,b),a=-0.1,b=0.1)
    # U_2 = torch.nn.init.uniform_(torch.empty(b,c),a=-0.1,b=0.1)
    # U_3 = torch.nn.init.uniform_(torch.empty(c,d),a=-0.1,b=0.1)
    # B = torch.nn.init.uniform_(torch.empty(d,d),a=-0.1,b=0.1)s
    U_1 = nn.ReLU()(U_1_origin.clone())
    U_2 = nn.ReLU()(U_2_origin.clone())
    # U_3 = nn.ReLU()(U_3_origin.clone())
    B = nn.ReLU()(B_origin.clone())
    B = (B+B.T)/2
    Y = torch.Tensor(Y)
    R = torch.Tensor(R)
    R_1 = torch.Tensor(R_1)
    h_B=0
    h_U_1=0
    h_U_2=0
    a = []
    b=[]
    while t < step: 
        T_0 = R-torch.mul(Y,net([U_1,U_2,B,U_2.T,U_1.T]))
        list1=[U_1,U_2]
        B_agd = -2*agd_B(T_0,list1)#negative
        # B_agd_2 = 2*agd_B(T_0,list1)
        mask_B = (B <= 0)
        h_B,B_origin = relu(B_origin,h_B,lr,mu,mask_B,B_agd)
        B = nn.ReLU()(B_origin)
        
        
        T_0 = R-torch.mul(Y,net([U_1,U_2,B,U_2.T,U_1.T]))
        list1=[U_1,U_2,B]
        
        U_1_agd = -4*agd_U(1,T_0,list1)
        mask_U_1 = (U_1 <= 0)
        h_U_1,U_1_origin = relu(U_1_origin,h_U_1,lr,mu,mask_U_1,U_1_agd)
        U_1 = nn.ReLU()(U_1_origin)
        
        
        
        T_0 = R-torch.mul(Y,net([U_1,U_2,B,U_2.T,U_1.T]))
        list1=[U_1,U_2,B]
        U_2_agd = -4*agd_U(2,T_0,list1)
        mask_U_2 = (U_2 <= 0)
        h_U_2,U_2 = relu(U_2,h_U_2,lr,mu,mask_U_2,U_2_agd) 
        U_2 = nn.ReLU()(U_2)
         
        t+=1
        LF = net([U_1,U_2,B,U_2.T,U_1.T])
        loss_matrix = LF-R_1
        loss_matrix[torch.isnan(R_1)]=0
        MAE = np.linalg.norm(loss_matrix,ord=1)/number
        RMSE = np.sqrt(np.linalg.norm(loss_matrix, ord=2)/number)
        
        a.append(MAE)
        b.append(RMSE)
        if len(a) == 1:
            pass
        elif abs(a[-1]+b[-1]-b[-2]-a[-2])< 1e-10:
            print(t)
            print("MAE:%.10f, RMSE:%.10f" % (MAE,RMSE))
            return MAE, RMSE,a,b
    print("MAE:%.10f, RMSE:%.10f" % (MAE,RMSE))
    return MAE, RMSE,a,b
# R_true = pd.read_pickle('/home/wujinrong/Desktop/Paper_A/数据集/数据集2/矩阵数据集合/data2')
# R_1 = pd.read_pickle('/home/wujinrong/Desktop/Paper_A/数据集/数据集2/测试数据集/缺失87%/R')
# print('R_1 \n',R_1.shape,R_1,'\n','R_true \n',R_true.shape,'\n',R_true)

# function(R_1,R_true,800, 200, 0.01, 500) 
# MAE:0.0019542990, RMSE:0.0159380077