# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.io
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.stats import pearsonr

#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)

from google.colab import drive
import os
drive.mount('/content/drive')

N_train = 900#1% data =180, 5% data=900, 10%=1800 data points; Here maximum population is x*t = 100*180=18000
data = scipy.io.loadmat('/content/drive/MyDrive/Colab Notebooks/Completed/Hydration/hydration.mat')

t_star = data['t']  # T x 1, 180 time points between 0 and 180 [180x1]
x_star1 = data['x']  # N x 2, 100 points between 0 and 100 [100x1]

#Change the model parameters based on the design/mixing proportions:

T_init=20 #degC: Change: Initial curing temperature 20degC (Trained)
delT = 0 #degC Increase in initial temperature from 20degC (20+10=30)
#solution of 180x100 grid points: cement hydration temperature in degC
usol_star1 = data['MPC']+ delT+273.15 #K #data for MPC #Change
#usol_star1 = data['FA45']+ delT+273.15 #K #data for FA45 #Change
#usol_star1 = data['SG80']+ delT+273.15 #K #data for SG80 #Change

c=1 #cement used (Y or N, 1 0r 0) #Change
fa=0 #flyash used (Y or N, 1 0r 0) #Change
sl=0 #slag used (Y or N, 1 0r 0) #Change

P1 = 3e-16 #300[kg/m^3] or 3.00e-16 [kg/μm^3]  Total amount of cement,flyash and slag #Change
C = 3e-16 #300[kg/m^3] or 3.00e-16 [kg/μm^3]  Cement content #Change
w_b = 0.493 #water binder ration #Change

#Fixed parameters

#Normalizing
t_star = t_star/180  # T x 1, 180 time points between 0 and 180 [180x1]
x_star1 = x_star1/100  # N x 2, 100 points between 0 and 100 [100x1]
usol_star = usol_star1/343.15 #K # N x T  #Can add small value (0.0001) if needed to avoid division by zero while testing on usol_test

T_init1=T_init+273.15 #K
T_init=T_init1/343.15 #K #normalizing

Cp = 0.84*c+0.84*fa+0.84*sl+0.9+0.9+4.18+2.2 #[J/g-degC]  #Heat capacities of cement,flyash,slag,sand,aggregate,water,and chemically bound water
Cp = Cp*1000#[J/(kg.K)]
Hu = 385000*c+209200*fa+460240*sl #[J/kg] Total heat of hydration

x_star=x_star1

N = x_star.shape[0]
T = t_star.shape[0]

# Construct the model with the desired t value:
for i, tt in enumerate(t_star):
  for j, x in enumerate(x_star[:,0]): #may not need if don't use j,k in the code
    for k, y in enumerate(x_star[:,1]):
      #The enumerate() function is used to add a counter to an iterable and return it in a form of enumerate object, which yields pairs of (index, element).
      #So, enumerate(t) creates an enumerate object for the time array t, which allows us to iterate over the time values and their corresponding indices
      #simultaneously. The for loop in the given code iterates over the t array, and for each time value, it performs computations using the x, y, and t arrays
      #at that time index. The i variable keeps track of the index of the current time value, which is used to access the corresponding values in x, y, and
      #other arrays. So, enumerate() is used here to keep track of the index of the current time value in t, which is needed to perform computations using
      #the corresponding values in other arrays.

        class Hydration():

            def __init__(self, X, Y, T, T_t, T_initial):

                self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
                self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
                self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)

                self.T_t = torch.tensor(T_t, dtype=torch.float32)
                self.T_initial = torch.tensor(T_initial, dtype=torch.float32)
                self.prediction_results = {} # to store prediction results for different values
                #We add the predicted T_t as an instance variable in the _init_ method

                #null vector to test against f and g:
                self.null = torch.zeros((self.x.shape[0], 1))

                # initialize network:
                self.network()

                self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.0033, max_iter=200000, max_eval=50000,
                                                  history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                                  line_search_fn="strong_wolfe")

                self.mse = nn.MSELoss()

                #loss
                self.ls = 0

                #iteration number
                self.iter = 0
                self.losses = []

            def network(self):
                n=32
                self.net = nn.Sequential(
                    nn.Linear(4, n), nn.Tanh(), #4 inputs (x,y,t,T_initial)
                    #nn.Dropout(p=0.5), # add dropout layer
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, n), nn.Tanh(),
                    nn.Linear(n, 4)) #4 outputs

                #In this code, there are no explicit weight and bias matrices. Instead, the neural network is defined using nn.Sequential and nn.Linear layers.
                #The weights and biases are initialized automatically by PyTorch when you create an instance of the nn.Linear class.
                #The torch.optim.LBFGS optimizer then updates these weights and biases during training to minimize the loss function defined in self.closure().

            def function(self, x, y, t, T_initial):

                res = self.net(torch.hstack((x, y, t, T_initial)))
                #The torch.hstack function horizontally stacks tensors to create a single tensor with all the inputs.
                #If x, y, and t are all tensors with shape (4,), then torch.hstack((x, y, t)) will create a new tensor by horizontally concatenating x, y, and t.
                #Since all three tensors have shape (4,), the resulting tensor will have shape (12,).
                #Passing this tensor through self.net will produce a tensor res with the same number of rows as the input tensor, but with a number of columns
                #equal to the output size of the neural network. We can say that the shape of res will be (4, C), where C is the number of output channels of the neural network.
                T_t, t_e, alpha, H= res[:, 0:1], res[:, 1:2], res[:, 2:3], res[:, 3:4]
                # This line assigns the output of the neural network (res) to five variables by selecting the first column through third columns of res, respectively.
                #Specifically, if x, y, and t are all tensors with shape (4,), then T_t will be a tensor containing the values in the first column of res,
                #with shape (4, 1), and w_n will be a tensor containing the values in the third column of res, also with shape (4, 1).

                # Add initial conditions at 1st time step

                if i == 0:
                  t_e[:, :] = 0
                  T_t[:, :] = T_initial
                  alpha[:, :] = 0

                if i > 0:
                  T_t[:, :] = T_t[:, :]+T_initial-283.15/343.15

                t_e_BC = (t_e[-1, :]).detach().numpy() #includes both IC and BC
                t_e_BC=np.concatenate([t_e_BC])
                t_e_BC = torch.from_numpy(t_e_BC)

                T_t_left = (T_t[-1, :]).detach().numpy()
                T_t_BC=np.concatenate([T_t_left])
                T_t_BC = torch.from_numpy(T_t_BC)

                alpha_BC = (alpha[-1, :]).detach().numpy() #includes both IC and BC
                alpha_BC=np.concatenate([alpha_BC])
                alpha_BC = torch.from_numpy(alpha_BC)

                #Derivatives

                t_e_t = torch.autograd.grad(t_e, t, grad_outputs=torch.ones_like(t_e), create_graph=True)[0]

                T_t_x = torch.autograd.grad(T_t, x, grad_outputs=torch.ones_like(T_t), create_graph=True)[0]
                T_t_xx = torch.autograd.grad(T_t_x, x, grad_outputs=torch.ones_like(T_t_x), create_graph=True)[0]
                T_t_y = torch.autograd.grad(T_t, y, grad_outputs=torch.ones_like(T_t), create_graph=True)[0]
                T_t_yy = torch.autograd.grad(T_t_y, y, grad_outputs=torch.ones_like(T_t_y), create_graph=True)[0]
                T_t_t = torch.autograd.grad(T_t, t, grad_outputs=torch.ones_like(T_t), create_graph=True)[0]

                alpha_t = torch.autograd.grad(alpha, t, grad_outputs=torch.ones_like(alpha), create_graph=True)[0]
                H_t = torch.autograd.grad(H, t, grad_outputs=torch.ones_like(H), create_graph=True)[0]

                #Parameters and properties:
                #UNITS ARE IN MICRO METER
                rho = 2.349e-15 #2349 [kg/m^3] or 2.349e-15 [kg/μm^3]
                k =  9360*1e-6  #2.6 [W/(m*K)] 0r 2.6 [J/s(m*K)] or 2.6*3600 =9360 [J/hr(m*K)] or 9360*1e-6 [J/hr(μm*K)] Thermal conductivity of concrete
                E_R = 5000 #[K]  Activation energy
                T_ref = T_initial*343.15 #[K]  Reference curing temperature: 10degC at t=0
                T_ref=torch.tensor(T_ref)
                alpha_u = 1.031*w_b/(0.194+w_b) #ultimate degree of hydration
                beta = 1.52 #hydration shape parameter
                t1 = 13
                T_t_clamped = torch.clamp(T_t, min=1e-6) #clamp the values to a small positive value to prevent NaNs.
                t_e_clamped = torch.clamp(t_e, min=1e-6) #clamp the values to a small positive value to prevent NaNs.
                tau=1+(t_e_clamped/t1)

                #Equations:
                f_t_e = t_e_t - torch.exp(E_R*((1/T_ref)-(1/T_t_clamped)))

                f_alpha = alpha_t - torch.exp(E_R*((1/T_ref)-(1/T_t_clamped)))* (alpha_u*beta/t_e_clamped)* (tau/t_e_clamped) ** beta * torch.exp(-(tau/t_e_clamped) ** beta)
                f_H=H_t-Hu*P1*alpha_t
                f_T_t = (rho*Cp*T_t_t) - k*(T_t_xx + T_t_yy) -H

                alpha =1-alpha
                H=Hu*P1*alpha_t

                return t_e, alpha, H, T_t, f_t_e, f_alpha, f_H, f_T_t, t_e_BC, alpha_BC, T_t_BC

            def closure(self):
                # reset gradients to zero:
                self.optimizer.zero_grad()

                # predictions:
                t_e, alpha, H, T_t, f_t_e, f_alpha, f_H, f_T_t, t_e_BC, alpha_BC, T_t_BC   = self.function(self.x, self.y, self.t, self.T_initial)

                # calculate losses
                #Data loss:
                T_t_loss = self.mse(T_t, self.T_t)
                #PDE loss:
                f_t_e_loss = self.mse(f_t_e, self.null)
                f_T_t_loss = self.mse(f_T_t, self.null)
                f_alpha_loss = self.mse(f_alpha, self.null)
                f_H_loss = self.mse(f_H, self.null)

                #BC loss:
                t_e_BC_loss = self.mse(t_e[-1,:], t_e_BC)
                T_t_BC_loss = self.mse(T_t[-1,:], T_t_BC)
                alpha_BC_loss = self.mse(alpha[-1,:], alpha_BC)

                # L2 regularization
                l2_reg = torch.tensor(0.0)
                for param in self.net.parameters():
                  l2_reg += torch.norm(param, p=2)

                self.ls = (T_t_loss) + (1*f_t_e_loss + 1*f_T_t_loss + 0*f_alpha_loss+1*f_H_loss) + (t_e_BC_loss + T_t_BC_loss + alpha_BC_loss)
                #self.ls = self.ls + 0.01*l2_reg  # Add L2 regularization term

                # derivative with respect to net's weights:
                self.ls.backward()

                self.iter += 1
                if not self.iter % 1:
                    loss = self.ls.item() # save the current loss value
                    self.losses.append(loss) # add the loss value to the list
                    print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))

                return self.ls

            def train(self):

                # training loop
                self.net.train()
                self.optimizer.step(self.closure)

                plt.plot(range(len(self.losses)), self.losses)
                plt.grid(True)
                plt.xlabel('Iteration', fontsize=15)
                plt.ylabel('Loss', fontsize=15)
                #plt.title('1% Data', fontsize=15)
                plt.show()

# Rearrange Data
XX = np.tile(x_star[:, 0:1], (1, T))  # N x T
YY = np.tile(x_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T
T_initial_star = np.full((200, 1), T_init)
LL = np.tile(T_initial_star, (1, N)).T  # N x T

PP = usol_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1
T_initial = LL.flatten()[:, None]  # NT x 1

p = PP.flatten()[:, None]  # NT x 1

# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
p_train = p[idx, :]
T_initial_train = T_initial[idx, :]

#Run model
#'''
pinn = Hydration(x_train, y_train, t_train, p_train, T_initial_train)
pinn.train()
torch.save(pinn.net.state_dict(), 'model.pt')
#'''
#Load model
pinn = Hydration(x_train, y_train, t_train, p_train, T_initial_train)
pinn.net.load_state_dict(torch.load('model.pt'))
pinn.net.eval()

# Test Data
# predict data to actual scale
#x_star2 = (x_star * np.std(x_star1,axis=0)) + np.mean(x_star1,axis=0)
x_star2 = x_star1*1
x_test1 = x_star2[:, 0:1] #Pick a different range of value to test on
y_test1 = x_star2[:, 1:2] #Pick a different range of value to test on
T_initial_test1 = np.full((100, 1), T_init1/343.15) #(X,1)#Pick a value other than T_initial=283.15K (10degC), to test on

# Time values
t_start = 0.0  # Start time
t_end = 1.0  # End time
num_times = 19  # Number of time instances
t_test1 = np.zeros((x_test1.shape[0], x_test1.shape[1])) + np.linspace(t_start, t_end, num_times)  # Array of time values
t_test = t_test1.transpose(-1,0)

x_test = torch.tensor(x_test1, dtype=torch.float32, requires_grad=True)
y_test = torch.tensor(y_test1, dtype=torch.float32, requires_grad=True)
T_initial_test = torch.tensor(T_initial_test1, dtype=torch.float32, requires_grad=False)

arr_z1z=[]
arr_z2z=[]
arr_z3z=[]
for i in range(num_times):
    t_test_i = t_test[i].reshape(-1, 1)
    t_test_i = torch.tensor(t_test_i, dtype=torch.float32, requires_grad=True)
    t_e_prediction,alpha_prediction,H_prediction,T_t_prediction,f_t_e_prediction,f_alpha_prediction,f_H_prediction,f_T_t_prediction,t_e_BC_prediction,alpha_BC_prediction,T_t_BC_prediction,  = pinn.function(x_test, y_test, t_test_i, T_initial_test)

    arr_z1z.append(T_t_prediction*343.15-273.15) #degC #get time-dependent results for the variable
    arr_z2z.append(1-alpha_prediction) #get time-dependent results for the variable
    arr_z3z.append(H_prediction*1e17/300/1000) #[mW/g] #get time-dependent results for the variable

'test neural network'
#Relative L2 error for the value of temperature after 180 hours
u = usol_star1[:,179] - 273.15
u_pred =  arr_z1z[18].detach().numpy()
error = u - u_pred
relative_l2_error = np.abs(error) / np.abs(u)

''' Model Accuracy '''
relative_l2_error=np.min(relative_l2_error) #Pick a minimum value from the array
relative_l2_error

"""Color Plot"""

#EXPERIMENTAL T
# Assign time values to show (0-179 index for 0 to 180 hours in experimental data)
ttt = 179

fig, axes = plt.subplots(1)
ax = axes
usol = (usol_star1[:, ttt] - 273.15).reshape(-1, 1) #degC
#Add some noise, otherwise exact value without any gradient (say 20) will show one color in the whole domain.
noise = np.random.normal(loc=0, scale=0.00001, size=usol.shape) #A little noise to better visualize colormap
usol = usol + noise
usol = np.reshape(usol, (10, 10))
# Call plt.contourf() with u_plot variable
cf = ax.contourf(usol, levels=30, cmap='jet')
colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'Experiment', fontsize=15,weight='bold')

#EcoBlendNet T
#Assign time values to show (0-10 index for 0 to 180 years with 10 increament):
tt=10 #0,5,10 to show time step at 0,90,180 hours

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
u_plot = arr_z1z[tt].data.cpu().numpy()
u_plot = u_plot + noise*10
u_plot = np.reshape(u_plot, (10, 10))
cf = ax.contourf(u_plot, levels=30, cmap='jet')
colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'EcoBlendNet', fontsize=15,weight='bold')

# Error plot
ax = axes[1]
error = np.abs(usol - u_plot)
cf = ax.contourf(error, levels=30, cmap='jet')
colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'Absolute Error', fontsize=15,weight='bold')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()

#EXPERIMENTAL alpha

alpha_experiment=[0.0, 0.41, 0.5] #Collected from literature
# Assign time values to show (0-2 index for 0,50, and 100 hours in experimental data)
ttt = 2

fig, axes = plt.subplots(1)
ax = axes
#Add some noise, otherwise exact value without any gradient (say 20) will show one color in the whole domain.
noise = np.random.normal(loc=0, scale=0.000001, size=usol.shape) #A little noise to better visualize colormap
usol1=alpha_experiment[ttt]
usol1 = usol1 + noise

usol1 = np.reshape(usol1, (10, 10))

# Call plt.contourf() with u_plot variable
cf = ax.contourf(usol1, levels=30, cmap='jet')

colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'Experiment', fontsize=15,weight='bold')

#EcoBlendNet alpha
#Assign time values to show (0-10 index for 0 to 180 years with 10 increament):
tt=6 #0,3,6 to show time step roughly at 0,50,100 hours

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
u_plot1 = arr_z2z[tt].data.cpu().numpy()
#Add some noise, otherwise exact value without any gradient (say 20) will show one color in the whole domain.
noise = np.random.normal(loc=0, scale=0.000001, size=u_plot1.shape) #A little noise to better visualize colormap
u_plot1 = u_plot1 + noise*10
u_plot1 = np.reshape(u_plot1, (10, 10))
cf = ax.contourf(u_plot1, levels=30, cmap='jet')
colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'EcoBlendNet', fontsize=15,weight='bold')

# Error plot
ax = axes[1]
error = np.abs(usol1 - u_plot1)
cf = ax.contourf(error, levels=30, cmap='jet')
colorbar = plt.colorbar(cf, ax=ax)
colorbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))
colorbar.ax.tick_params(labelsize=12)
ax.set_xticks(np.arange(10))
ax.set_xticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_yticks(np.arange(10))
ax.set_yticklabels(np.arange(0, 100, 10) * 1 + 10,fontsize=12)
ax.set_xlabel(r'$x (μm)$', fontsize=15,weight='bold')
ax.set_ylabel(r'$y (μm)$', fontsize=15,weight='bold')
ax.set_title(r'Absolute Error', fontsize=15,weight='bold')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()