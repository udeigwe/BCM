from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from scipy.integrate import quad

class BCMneuron:
    def __init__(self, lgn_l, lgn_r, p=2, eta=0.005, tau=1000, c0=50, condition='NR'):
        self.lgn_l = lgn_l #Number of LGN-cortical input fibers from left eye
        self.lgn_r = lgn_r  # Number of LGN-cortical input fibers from right eye
        self.np = np #Numebr of training patterns
        self.p = p #Nonlinearity in the definition of theta, here, theta is u
        self.eta = eta #Modification step size
        self.tau = tau #Time constant in the definition of theta
        self.c0 = c0 #Normalization constant in the definition of theta
        self.condition = condition
            #NR = normal rearing
            #MD = monocular deprivation
            #RS = reverse suture
            #ST = strabismus
            #BD = binocular deprivation
        self.ml = 0.15*np.random.rand(1,self.lgn_l) #weights initialized to numbers btw 0 and 0.15
        self.mr = 0.15*np.random.rand(1, self.lgn_r)  # weights initialized to numbers btw 0 and 0.15
        self.theta = np.random.rand()

    def train (self, da_l,da_r,ds_l, ds_r, nl=0, nr=0, iterations=1000):
        # da = actual pattern; ds = spontaneous; n = non-lgn noise
        if (self.lgn_l != np.shape(da_l)[1]) or (self.lgn_r != np.shape(da_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        elif (self.lgn_l != np.shape(ds_l)[1]) or (self.lgn_r != np.shape(ds_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        else:
            for i in range(iterations):
                k  = np.random.randint(np.shape(da_l)[0])
                ca = sum(self.ml*da_l[k]) + sum(self.mr*da_r[k])
                cs = sum(self.ml*ds_l[k]) + sum(self.mr*ds_r[k])
                cn = sum(self.ml*nl[k]) + sum(self.mr*nr[k])
                c = ca + cs + cn
                phi = c*(c-self.theta)
                self.ml = self.ml + self.eta*phi*dl #update left weights
                self.mr = self.mr + self.eta*phi*dr #update right weights











        for i in range(iterations):


    def train_NR(self,patterns_le,patterns_re,iterations):
        patterns = np.array(patterns)
        num_patterns = np.shape(patterns)[0]











    def response_le(self):

    def response_re(self):

    def res






