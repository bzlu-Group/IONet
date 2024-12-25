# %%
from sklearn import gaussian_process as gp
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os 

#### basic  parameters
length_scale = 0.25          ### control GRF
features = 1000      
sensors_num = 100  
tol = 1e-5

Remark = 'Test'              ### generate test data 
#Remark = 'Train'            ### generate training data

if Remark == 'Train':
    NNN=10   
    test_save = False 
    train_save = True
    samples_num = 10000  ### train samples 10000

if Remark == 'Test':
    NNN=1
    test_save =  True
    train_save = False 
    samples_num = 1000  ### test samples 1000

############################### mesh ##############################
N = 998
xmin,xmax = 0,1
h = (xmax-xmin)/(N+1)
mesh = np.linspace(xmin, xmax, N+2,endpoint=True)  

gamma = 0.5  # interface location


def interface_dirich():
    return 1


def interface_numan():
    return 0.


def boundary_condition_l():
    return 1


def boundary_condition_r():
    return 0.


def rhs(x):
    return np.zeros_like(x)


def weight_order0(alpha,x_array):
    a=[]
    # Taylor expansion
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([1,0,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)
    return w.reshape(-1,)


def weight_order1(alpha,x_array):
    a=[]
    a.append([1,1,1])
    a.append(x_array-alpha)
    a.append((x_array-alpha)**2)
    a=np.array(a)
    b=np.array([0,1,0]).reshape(-1,1)
    w=np.dot(np.linalg.inv(a), b)
    return w.reshape(-1,)


def fi_and_fiadd1(w01,w02,w11,w12,k1,k2):
    """
    coef local: [ui-1,ui,u+1]
    w01,w11: ui-1, ui, ui+1
    w02,w12: ui, ui+1, ui+2
    """
    a=np.array([[w02[0],-w01[2]],[k2*w12[0],-k1*w11[2]]]) 
    a_inv=np.linalg.inv(a)
    b=np.array([[w01[0],w01[1],-w02[1],-w02[2],1,0],[k1*w11[0],k1*w11[1],-k2*w12[1],-k2*w12[2],0,1]]) 
   
    return a_inv@b


def modify_matrix(A,F,mesh, gamma, record, coef, coefx, k1, k2):
    """
    alpha in i and i+1
    
    A_local@[ui-1,ui,ui+2,ui+2,[u],[un]].T=[fi,fi+1].T
    k1, k2: approximate coefficient value in 
    """

    i=record[0]-1                      
    x_1=mesh[record[0]-1:record[1]+1] # xi-1,xi,xi+1
    x_2=mesh[record[0]:record[1]+2]   # xi, xi+1,xi+2
    w01= weight_order0(gamma,x_1)     # fi+1
    w02= weight_order0(gamma,x_2)     # fi

    w11= weight_order1(gamma,x_1)
    w12= weight_order1(gamma,x_2)

    A_local=fi_and_fiadd1(w01, w02, w11, w12, k1, k2)

     
    A[i,i+1] += coef[i+1]                       
    A[i,i-1:i+3]-= A_local[1,:4]*coef[i+1]

    A[i+1,i] += coef[i+2]                         
    A[i+1,i-1:i+3] -= A_local[0,:4]*coef[i+2]
    inter_condi=np.array([interface_dirich(),interface_numan()]).reshape(-1,1)

    F[i] += (A_local[:,4:]@inter_condi)[1][0]*coef[i+1]
    F[i+1] += (A_local[:,4:]@inter_condi)[0][0]*coef[i+2]
 
 
    A[i,i+1] += h/2*coefx[i+1]                     
    A[i,i-1:i+3] -= h/2* A_local[1,:4]*coefx[i+1]

    A[i+1,i] -= h/2*coefx[i+2]
    A[i+1,i-1:i+3] += h/2*A_local[0,:4]*coefx[i+2]       

    F[i] += h/2*(A_local[:,4:]@inter_condi)[1][0]*coefx[i+1]
    F[i+1] -= h/2* (A_local[:,4:]@inter_condi)[0][0]*coefx[i+2]  

    return A,F



def deal_with_mesh(coef_l, coef_m,  gamma,  mesh=mesh,  tol=tol):
    """
    input: mesh, alpha, k1, k2
    output:record and coef (two np.array)
    """   
    coef=np.zeros(N+2)
    coefx=np.zeros(N+2)
 
    index1=np.argmin(abs(mesh-gamma))
    if mesh[index1]<=gamma and mesh[index1+1]>gamma:
        coef[:index1+1]=coef_l(mesh[:index1+1])
        coefx[:index1+1] =(coef_l(mesh[:index1+1]+tol)-coef_l(mesh[:index1+1]-tol))/2/tol
        coef[index1+1:]=coef_m(mesh[index1+1:])
        coefx[index1+1:] =(coef_m(mesh[index1+1:]+tol)-coef_m(mesh[index1+1:]-tol))/2/tol
        record1= [index1,index1+1]
        wl = weight_order0(gamma, mesh[index1-1:index1+2])
        wr = weight_order0(gamma, mesh[index1:index1+3])
        appro_coef_g1_l = wl@coef_l(mesh[index1-1:index1+2])
        appro_coef_g1_r = wr@coef_m(mesh[index1:index1+3])
            
    elif mesh[index1]>gamma and mesh[index1-1]<=gamma:
        coef[:index1]=coef_l(mesh[:index1])
        coefx[:index1] =(coef_l(mesh[:index1]+tol)-coef_l(mesh[:index1]-tol))/2/tol
        coef[index1:]=coef_m(mesh[index1:])
        coefx[index1:] =(coef_m(mesh[index1:]+tol)-coef_m(mesh[index1:]-tol))/2/tol
        record1= [index1-1,index1]
        wl = weight_order0(gamma, mesh[index1-2:index1+1])
        wr = weight_order0(gamma, mesh[index1-1:index1+2])
        appro_coef_g1_l = wl@coef_l(mesh[index1-2:index1+1])
        appro_coef_g1_r = wr@coef_m(mesh[index1-1:index1+2]) 
        
    K=[appro_coef_g1_l, appro_coef_g1_r]
    return np.array(record1),  np.array(coef) ,np.array(coefx), K


def Generate_sensors(sensors_num, gamma):
 
    x_sensor=np.linspace(0, 1, num=sensors_num) 
    xl_sensor = x_sensor[np.where(x_sensor<=gamma)[0]]
    xr_sensor = x_sensor[np.where(x_sensor>gamma)[0]]
    
    return xl_sensor, xr_sensor


def generate_gps(length_scale, sample_num, features = features):
 
    x = np.linspace(-0.01, 1.01, num=features)[:, None]
    A = gp.kernels.RBF(length_scale=length_scale)(x)
    L = np.linalg.cholesky(A + 1e-13 * np.eye(features))
    aa=  L @ np.random.randn(features, sample_num)
    gps=(aa-np.min(aa, axis=0,keepdims=True)+1).transpose()

    return gps

 
gps_l = generate_gps(length_scale, samples_num)
gps_r = generate_gps(length_scale, samples_num)

u_pre_solution = []     
coef_sensor_value = []
coefx_sensor_value = []
sensors_location = np.linspace(0, 1, num=sensors_num)

 
appro_coef_g1_l = []
appro_coef_g1_r = []
 
xx_l = mesh[::NNN][np.where(mesh[::NNN]<=gamma)[0]]
xx_r = mesh[::NNN][np.where(mesh[::NNN]>gamma)[0]]

coef_train_x = []     
coefx_train_x = []  

xl_sensor, xr_sensor = Generate_sensors(sensors_num, gamma)

import time
t0=time.time()
for k in range(samples_num):
    if k%200==0:
        print(k)

    coef_l=interpolate.interp1d(np.linspace(-0.01, 1.01, num=features), gps_l[k], kind='cubic', copy=False, assume_sorted=True)
    coef_r=interpolate.interp1d(np.linspace(-0.01, 1.01, num=features), gps_r[k], kind='cubic', copy=False, assume_sorted=True) 
   
    record,  coef, coefx, K =  deal_with_mesh(coef_l, coef_r, gamma,  mesh=mesh, tol=tol)

    appro_coef_g1_l.append(K[0])
    appro_coef_g1_r.append(K[1])
 
    u_pre = np.zeros(N+2)
    u_pre[0]= boundary_condition_l()
    u_pre[-1]= boundary_condition_r()

    A1 = np.zeros((N, N))  
    for i in range(N):
        A1[i, i] = -2
        if i < N-1: A1[i, i+1] = 1
        if i > 0:   A1[i, i-1] = 1 
    A1=-np.diag(coef[1:-1])@A1

 
    A2 = np.zeros((N, N))  
    for i in range(N):
        if i < N-1: A2[i, i+1] = 1
        if i > 0:   A2[i, i-1] = -1 
    A2=-np.diag(coefx[1:-1])@A2*h/2
    A= A1+A2

 
    F = rhs(mesh[1:-1]).reshape(-1,1)

     
    F[0] += u_pre[0]*coef[1]
    F[-1] += u_pre[-1]*coef[-2]

     
    F[0] -= u_pre[0]*coefx[1]*h/2
    F[-1] += u_pre[-1]*coefx[-2]*h/2

    
    A_new,F_new=modify_matrix(A,F,mesh, gamma, record, coef, coefx, K[0], K[1])

    u_pre[1:-1]=np.dot(np.linalg.inv(A_new), F_new).reshape(-1,)
    u_pre_solution.append(u_pre[::NNN])
    

     
    ### left 
    coef_left=coef_l(xl_sensor)
    ### right
    coef_right=coef_r(xr_sensor)
    coef_sensor_value.append(np.hstack((coef_left, coef_right)))
    
     
    coef_xl = coef_l(xx_l)
    coefx_xl = (coef_l(xx_l+tol)-coef_l(xx_l-tol))/2/tol
    coef_xr = coef_r(xx_r)
    coefx_xr = (coef_r(xx_r+tol)-coef_r(xx_r-tol))/2/tol
    
    coef_train_x.append(np.hstack((coef_xl, coef_xr)))
    coefx_train_x.append(np.hstack((coefx_xl, coefx_xr)))


# %%
data_file = 'data/'
print(os.getcwd())
if not os.path.isdir('./'+data_file): os.makedirs('./'+data_file)

np.savetxt(data_file+'/sensors_location.txt',np.vstack(sensors_location))
if train_save== True:
  
    np.savetxt(data_file+'/train_appro_coef_1l.txt', appro_coef_g1_l)
    np.savetxt(data_file+'/train_appro_coef_1r.txt', appro_coef_g1_r)
 
    np.savetxt(data_file+'/train_sensors_location.txt',sensors_location)
    np.savetxt(data_file+'/train_coef_sensor_value.txt',np.vstack(coef_sensor_value))
    
    np.savetxt(data_file+'/train_mesh.txt', mesh[::NNN])
    np.savetxt(data_file+'/train_u_solution.txt',  np.vstack(u_pre_solution))
    np.savetxt(data_file+'/train_coef_x.txt',      np.vstack(coef_train_x))
    np.savetxt(data_file+'/train_coefx_x.txt',     np.vstack(coefx_train_x))

if test_save== True:
    
    np.savetxt(data_file+'/test_mesh.txt',mesh[::NNN])
    np.savetxt(data_file+'/test_coef_sensor_value.txt',np.vstack(coef_sensor_value))
    np.savetxt(data_file+'/test_u_solution.txt',np.vstack(u_pre_solution))





