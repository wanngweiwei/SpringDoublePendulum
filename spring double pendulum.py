# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:49:08 2023

@author: Www
"""

# this is a program to analyse the equilibrium and stability of spring double pendulum

import numpy as np
import math
import scipy.integrate as si #引入积分用的函数
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from numpy import array,mat,sin,cos,exp
from scipy.optimize import root
N=2
m=1
g=9.8
k=1
L=1

def sdp(xy):

    x1=xy[0]
    y1=xy[1]
    x2=xy[2]
    y2=xy[3]

    fx1=k*(np.sqrt(x1**2+y1**2)-L)*(-x1)/np.sqrt(x1**2+y1**2)-k*(np.sqrt((x1-x2)**2+(y1-y2)**2)-L)*(x1-x2)/np.sqrt((x1-x2)**2+(y1-y2)**2)
    fy1=k*(np.sqrt(x1**2+y1**2)-L)*(-y1)/np.sqrt(x1**2+y1**2)-k*(np.sqrt((x1-x2)**2+(y1-y2)**2)-L)*(y1-y2)/np.sqrt((x1-x2)**2+(y1-y2)**2)-m*g
    fx2=k*(np.sqrt((x1-x2)**2+(y1-y2)**2)-L)*(x1-x2)/np.sqrt((x1-x2)**2+(y1-y2)**2)
    fy2=k*(np.sqrt((x1-x2)**2+(y1-y2)**2)-L)*(y1-y2)/np.sqrt((x1-x2)**2+(y1-y2)**2)-m*g

    F=[fx1,fy1,fx2,fy2]

    return F


jie=root(sdp,[0,-1,0,-2])
print(jie)

x1=jie.x[0]
y1=jie.x[1]
x2=jie.x[2]
y2=jie.x[3]




yakebi=np.zeros((4*N+1,4*N+1))
yakebi[1,5]=1
yakebi[2,6]=1
yakebi[3,7]=1
yakebi[4,8]=1

yakebi[5,1]= ((x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)*((-2*x1**2 - 2*y1**2)*np.sqrt(x1**2 + y1**2) + L*y1**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(x1**2 + y1**2)**(3/2)*(-y2 + y1)**2)*k/((x1**2 + y1**2)**(3/2)*(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2))
yakebi[5,2]=-(x1*y1*(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + (x1**2 + y1**2)**(3/2)*(-y2 + y1)*(-x2 + x1))*k*L/((x1**2 + y1**2)**(3/2)*(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2))
yakebi[5,3]=-((-y1**2 + 2*y2*y1 - y2**2 - (-x2 + x1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-y2 + y1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[5,4]=k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[6,1]=-(x1*y1*(x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + (x1**2 + y1**2)**(3/2)*(-y2 + y1)*(-x2 + x1))*k*L/((x1**2 + y1**2)**(3/2)*(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2))
yakebi[6,2]= ((x1**2 - 2*x1*x2 + x2**2 + y1**2 - 2*y1*y2 + y2**2)*((-2*x1**2 - 2*y1**2)*np.sqrt(x1**2 + y1**2) + L*x1**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(x1**2 + y1**2)**(3/2)*(-x2 + x1)**2)*k/((x1**2 + y1**2)**(3/2)*(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2))
yakebi[6,3]=k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[6,4]=-((-x1**2 + 2*x2*x1 - x2**2 - (-y2 + y1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-x2 + x1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[7,1]= -((-y1**2 + 2*y2*y1 - y2**2 - (-x2 + x1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-y2 + y1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[7,2]=k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[7,3]=((-y1**2 + 2*y2*y1 - y2**2 - (-x2 + x1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-y2 + y1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[7,4]=-k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[8,1]= k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[8,2]=-((-x1**2 + 2*x2*x1 - x2**2 - (-y2 + y1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-x2 + x1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[8,3]=-k*(-x2 + x1)*(-y2 + y1)*L/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)
yakebi[8,4]=((-x1**2 + 2*x2*x1 - x2**2 - (-y2 + y1)**2)*np.sqrt(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2) + L*(-x2 + x1)**2)*k/(x1**2 - 2*x2*x1 + x2**2 + (-y2 + y1)**2)**(3/2)


print(yakebi)


tezhengzhi=np.linalg.eig(yakebi)[0]
print('雅克比矩阵的特征值为',tezhengzhi)
