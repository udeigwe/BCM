# TTX injected in the ipsilateral eye (here, right eye);
# TTX reduces inhibition to neuron coming from an inhibitory neuron


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
import matplotlib.pyplot as plt
# hashtag hashtaggers

class BCMNeuron:
    def __init__(self, lgn_l, lgn_r, patterns, p=2, tau_w=10, tau_theta=1, dt=0.01, eps=0,
                 c0=50, condition='NR', iterations=60000, ttx_point=0.50, tau_ttx=300,
                 we=1, wi=0.09):
        self.lgn_l = lgn_l  # Number of LGN-cortical input fibers from left eye
        self.lgn_r = lgn_r  # Number of LGN-cortical input fibers from right eye
        self.patterns = patterns  # Number of training patterns
        self.p = p  # Nonlinearity in the definition of theta
        self.tau_w = tau_w  # Modification step size
        self.tau_theta = tau_theta  # Time constant in the definition of theta
        self.dt = dt  # simulation mesh size
        self.eps = eps;  # regularization constant for weights
        self.c0 = c0  # Normalization constant in the definition of theta
        self.iterations = iterations
        self.ttx_in = np.floor(ttx_point * iterations)  # time of ttx injection
        self.tau_ttx = tau_ttx
        self.condition = condition
        #       NR = normal rearing,
        #       MD = monocular deprivation,
        #       RS = reverse suture,
        #       ST = strabismus,
        #       BD = binocular deprivation
        #       TTX_decay = TTX injected in the ipsilatral eye (here, right eye); TTX decays
        #       TTX_inh = TTX injected in the ipsilateral eye (here, right eye); TTX reduces inhibition to neuron

        ### variables for excitatory (present) BCM neuron
        self.ml = 0.15 * np.random.rand(1, self.lgn_l)  # weights initialized to numbers btw 0 and 0.15
        self.mr = 0.15 * np.random.rand(1, self.lgn_r)  # weights initialized to numbers btw 0 and 0.15
        self.weights_l = np.random.rand(1, self.lgn_l)  # record matrix; each row is a weight vector for the iteration
        self.weights_r = np.random.rand(1, self.lgn_r, )  # record matrix; each row is a weight vector for the iteration
        self.theta = 0.15 * np.random.rand()  # sliding threshold initialized to a number btw 0 and 0.15
        self.thresholds = self.theta  # record array of thresholds
        self.tuning_curve_l = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration
        self.tuning_curve_r = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration

        ### variable for inhibitory neuron
        self.mli = 0.15 * np.random.rand(1, self.lgn_l)  # weights initialized to numbers btw 0 and 0.15
        self.mri = 0.15 * np.random.rand(1, self.lgn_r)  # weights initialized to numbers btw 0 and 0.15
        self.weights_li = np.random.rand(1, self.lgn_l)  # record matrix; each row is a weight vector for the iteration
        self.weights_ri = np.random.rand(1,
                                         self.lgn_r, )  # record matrix; each row is a weight vector for the iteration
        self.thetai = 0.15 * np.random.rand()  # sliding threshold initialized to a number btw 0 and 0.15
        self.thresholdsi = self.thetai  # record array of thresholds
        self.tuning_curve_li = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration
        self.tuning_curve_ri = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration

        ### inhibition parameters
        self.we = we  # excitatory weight from E to I
        self.wi = wi  # Inhibitory weight from I to E

    def train(self, da_l, da_r, da_li, da_ri, ds_l=0, ds_r=0,
              ds_li=0, ds_ri=0):
        # da => actual pattern; ds => spontaneous; n => non-lgn noise
        # r => right, l => left, i => inhibitory

        a = 0.00001  # noise scaler
        # ttx_in = np.floor(0.5*self.iterations)  # time of ttx injection
        # ttx_gone =  np.floor(0.75*self.iterations) # time when ttx wears out
        # ttx_decay = 10

        if (self.lgn_l != np.shape(da_l)[1]) or (self.lgn_r != np.shape(da_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        elif (self.lgn_l != np.shape(ds_l)[1]) or (self.lgn_r != np.shape(ds_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        else:

            for i in range(self.iterations):

                t = i * self.dt
                k = np.random.randint(np.shape(da_l)[0])

                # generate noise
                nl = a * np.random.rand(self.patterns, self.lgn_l)
                nr = a * np.random.rand(self.patterns, self.lgn_r)
                nli = a * np.random.rand(self.patterns, self.lgn_l)
                nri = a * np.random.rand(self.patterns, self.lgn_r)

                # compute BCM neuron activity
                # ca = np.sum(self.ml * da_l[k]) + np.sum(self.mr * da_r[k])
                # cs = np.sum(self.ml * ds_l[k]) + np.sum(self.mr * ds_r[k])
                # cn = np.sum(self.ml * nl[k]) + np.sum(self.mr * nr[k])
                # c = ca - cs + cn

                # compute inhibitory neuron activity
                cai = np.sum(self.mli * da_li[k]) + np.sum(self.mri * da_ri[k])
                csi = np.sum(self.mli * ds_li[k]) + np.sum(self.mri * ds_ri[k])
                cni = np.sum(self.mli * nli[k]) + np.sum(self.mri * nri[k])
                ci = cai - csi + cni

                # nonlinear funtion of threshold and activity for inhibitory neuron
                phii = ci * (ci - self.thetai)
                ci = 0

                if i <= self.ttx_in:
                    # compute BCM neuron activity
                    ca = 0 * np.sum(self.ml * da_l[k]) + np.sum(self.mr * da_r[k])  # only right eye gets input
                    cs = 0 * np.sum(self.ml * ds_l[k]) + np.sum(self.mr * ds_r[k])  # only right eye gets input
                    cn = np.sum(self.ml * nl[k]) + np.sum(self.mr * nr[k])  # both eyes get noise
                    c = ca - cs + cn
                    # Feedback loop
                    # ci = ci + self.we*c
                    # c = c - self.wi*ci
                    # nonlinear function of threshold and activity
                    phi = c * (c - self.theta)
                    # update left weights for BCM neuron
                    self.ml = self.ml + (1 / self.tau_w) * self.dt * (
                                phi * (0 * da_l[k] - 0 * ds_l[k] + nl[k]) - self.eps * self.ml)
                    # update right weights for BCM neuron
                    self.mr = self.mr + (1 / self.tau_w) * self.dt * (
                                phi * (da_r[k] - ds_r[k] + nr[k]) - self.eps * self.mr)
                    # Add responses to all orientation in the tuning curve matrix
                    self.tuning_curve_l = np.vstack(
                        (self.tuning_curve_l, np.sum((0 * da_l - 0 * ds_l + nl) * self.ml, axis=1)))
                    self.tuning_curve_r = np.vstack((self.tuning_curve_r, np.sum((da_r - ds_r + nr) * self.mr, axis=1)))
                else:
                    # ttx_factor = np.exp((i-ttx_gone)*np.heaviside(ttx_gone-i,0))
                    ttx_factor = (1 - np.exp(self.dt * (1 / self.tau_ttx) * (self.ttx_in - i)))
                    # ttx_factor = np.heaviside(i-60000,0)
                    # ttx_factor = 0
                    # print(ttx_factor)
                    # compute BCM neuron activity
                    ca = np.sum(self.ml * da_l[k]) + ttx_factor * np.sum(self.mr * da_r[k])
                    cs = np.sum(self.ml * ds_l[k]) + ttx_factor * np.sum(self.mr * ds_r[k])
                    cn = np.sum(self.ml * nl[k]) + ttx_factor * np.sum(self.mr * nr[k])
                    c = ca - cs + cn
                    # Feedback loop
                    ci = ci + self.we * c
                    c = c - self.wi * ci
                    # nonlinear funtion of threshold and activity
                    phi = c * (c - self.theta)
                    # update left weights for BCM neuron
                    self.ml = self.ml + (1 / self.tau_w) * self.dt * (
                                phi * (da_l[k] - ds_l[k] + nl[k]) - self.eps * self.ml)
                    # update right weights for BCM neuron
                    self.mr = self.mr + (1 / self.tau_w) * self.dt * (
                                phi * ttx_factor * (da_r[k] - ds_r[k] + nr[k]) - self.eps * self.mr)
                    # Add responses to all orientation in the tuning curve matrix
                    self.tuning_curve_l = np.vstack((self.tuning_curve_l, np.sum((da_l - ds_l + nl) * self.ml, axis=1)))
                    self.tuning_curve_r = np.vstack(
                        (self.tuning_curve_r, np.sum(ttx_factor * (da_r - ds_r + nr) * self.mr, axis=1)))

                # update left weights for inhibitory neuron
                self.mli = self.mli + (1 / self.tau_w) * self.dt * (
                            phii * (da_li[k] - ds_li[k] + nli[k]) - self.eps * self.mli)
                # update right weights for inhibitory neuron
                self.mri = self.mri + (1 / self.tau_w) * self.dt * (
                            phii * (da_ri[k] - ds_ri[k] + nri[k]) - self.eps * self.mri)

                # Add new weights to the weight matrices
                self.weights_l = np.vstack((self.weights_l, self.ml))
                self.weights_r = np.vstack((self.weights_r, self.mr))
                self.weights_li = np.vstack((self.weights_li, self.mli))
                self.weights_ri = np.vstack((self.weights_ri, self.mri))
                # We need to now update theta
                self.theta = self.theta + (1 / self.tau_theta) * self.dt * (ca ** self.p - self.theta)
                self.thetai = self.thetai + (1 / self.tau_theta) * self.dt * (ca ** self.p - self.thetai)
                # Add new theta to the vector of thresholds
                self.thresholds = np.hstack((self.thresholds, self.theta))
                self.thresholdsi = np.hstack((self.thresholdsi, self.thetai))
                # Add responses to all orientation in the tuning curve matrix for inhibitory neuron
                self.tuning_curve_li = np.vstack(
                    (self.tuning_curve_li, np.sum((da_li - ds_li + nli) * self.mli, axis=1)))
                self.tuning_curve_ri = np.vstack(
                    (self.tuning_curve_ri, np.sum((da_ri - ds_ri + nri) * self.mri, axis=1)))
        # Now delete the first entries (initializations) of the record matrices/vectors
        self.weights_l = np.delete(self.weights_l, 0, axis=0)
        self.weights_r = np.delete(self.weights_r, 0, axis=0)
        self.weights_li = np.delete(self.weights_li, 0, axis=0)
        self.weights_ri = np.delete(self.weights_ri, 0, axis=0)
        self.thresholds = np.delete(self.thresholds, 0)
        self.thresholdsi = np.delete(self.thresholdsi, 0)
        self.tuning_curve_l = np.delete(self.tuning_curve_l, 0, axis=0)
        self.tuning_curve_r = np.delete(self.tuning_curve_r, 0, axis=0)
        self.tuning_curve_li = np.delete(self.tuning_curve_li, 0, axis=0)
        self.tuning_curve_ri = np.delete(self.tuning_curve_ri, 0, axis=0)

        return self.weights_l, self.weights_r, self.thresholds, self.tuning_curve_l, self.tuning_curve_r

    # def train_nr(self):

    # def train_md(self):

    # def train_rs(self):

    # def train_bd(self):

    # def train_re(self):

    # def train_st(self):

    # def get_response(self):

    # def plot_responses(self):
    """
     To use
     neuron = BCMNeuron(...)
     neuron.train(...)
     neuron.plot_response()
    """

    # def plot_tuning_curve(self)
    """
     To use
     neuron = BCMNeuron(...)
     neuron.train(...)
     neuron.plot_tuning_curve()
    """

    def plot_weights(self):
        plt.figure(0)
        (row_l, col_l) = np.shape(self.weights_l)
        # print(row_l)
        # print (col_l)
        (row_r, col_r) = np.shape(self.weights_r)
        # print(row_r)
        # print (col_r)
        x = list(range(0, self.iterations))
        x = self.dt * np.array(x)
        figure, axis = plt.subplots(self.lgn_l, 2)

        for k in range(col_l):
            axis[k, 0].plot(x, self.weights_l[:, k])
            axis[k, 0].set_xlabel("Time")
            axis[k, 0].set_ylabel("Weight")
            axis[k, 0].set_title("Time vs Weight")
            #axis[k, 0]

        for k in range(col_r):
            axis[k, 1].plot(x, self.weights_r[:, k])
            axis[k, 1].set_xlabel("Time")
            axis[k, 1].set_ylabel("Weight")
            axis[k, 1].set_title("Time vs Weight")
        
        plt.savefig("/braintree/home/udeigwe/weights.png")
        # plt.legend()
        # plt.show()

    def plot_responses(self):
        plt.figure(1)
        (row_l, col_l) = np.shape(self.tuning_curve_l)
        (row_r, col_r) = np.shape(self.tuning_curve_r)
        x = list(range(0, self.iterations))
        x = self.dt * np.array(x)
        figure, axis = plt.subplots(self.patterns, 2)

        for k in range(col_l):
            axis[k, 0].plot(x, self.tuning_curve_l[:, k])
            axis[k, 0].set_xlabel("Time")
            axis[k, 0].set_ylabel("Response")
            axis[k, 0].set_title("Time vs Response")
            axis[k, 0].set_ylim([-0.2, 3.2])

        for k in range(col_r):
            axis[k, 1].plot(x, self.tuning_curve_r[:, k])
            axis[k, 1].set_xlabel("Time")
            axis[k, 1].set_ylabel("Response")
            axis[k, 1].set_title("Time vs Response")
            axis[k, 1].set_ylim([-0.2, 3.2])

        plt.savefig("resp.png")
        # plt.legend()
        #plt.show()

    def plot_2eyes(self):
        plt.figure(2)
        (row_l, col_l) = np.shape(self.tuning_curve_l)
        (row_r, col_r) = np.shape(self.tuning_curve_r)
        x = list(range(0, self.iterations))
        x = self.dt * np.array(x)
        figure, axis = plt.subplots(self.patterns)
        for k in range(self.patterns):
            axis[k].plot(x, self.tuning_curve_r[:, k])
            axis[k].plot(x, self.tuning_curve_l[:, k])
            axis[k].set_xlabel("Time")
            axis[k].set_ylabel("Response")
            axis[k].set_title("Time vs Response")
            axis[k].set_ylim([-0.2, 3.2])
        # plt.plot.xlim([0, eta*iterations])
        # axis[k].set_ylim([0, 2])
        plt.savefig("eyes.png")
        # axis.legend()



"""
def main():
    pat = 0.001*np.random.rand(3, 3) + np.identity(3)
    print(pat)
    neuron = BCMNeuron(lgn_l=3, lgn_r=3, patterns=3)
    wl,ws, th, tcl, tcr = neuron.train(da_l=pat, da_r=pat, ds_l=0*pat, ds_r=0*pat, nl=0*pat, nr=0*pat)
    neuron.plot_weights()

if __name__ == "__main__":
    main()
"""

pat = np.identity(3)
pat = (np.ones((3,3))-np.identity(3))
pati = pat
neuron = BCMNeuron(lgn_l=3, lgn_r=3, patterns=3)
wl, ws, th, tcl, tcr = neuron.train(da_l=pat, da_r=pat, \
                                    da_li=pati, da_ri=pati, \
                                    ds_l=0 * pat, ds_r=0 * pat, \
                                    ds_li=0 * pati, ds_ri=0 * pati) 
neuron.plot_weights()
neuron.plot_responses()
neuron.plot_2eyes()
plt.show()



