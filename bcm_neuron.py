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


class BCMNeuron:
    def __init__(self, lgn_l, lgn_r, patterns, p=2, eta=1 / 50, tau=100, c0=50, condition='NR',
                 iterations=10000):
        self.lgn_l = lgn_l  # Number of LGN-cortical input fibers from left eye
        self.lgn_r = lgn_r  # Number of LGN-cortical input fibers from right eye
        self.patterns = patterns  # Number of training patterns
        self.p = p  # Nonlinearity in the definition of theta
        self.eta = eta  # Modification step size
        self.tau = tau  # Time constant in the definition of theta
        self.c0 = c0  # Normalization constant in the definition of theta
        self.condition = condition
        self.iterations = iterations
        #       NR = normal rearing,
        #       MD = monocular deprivation,
        #       RS = reverse suture,
        #       ST = strabismus,
        #       BD = binocular deprivation
        self.ml = 0.15 * np.random.rand(1, self.lgn_l)  # weights initialized to numbers btw 0 and 0.15
        self.mr = 0.15 * np.random.rand(1, self.lgn_r)  # weights initialized to numbers btw 0 and 0.15
        self.weights_l = np.random.rand(1, self.lgn_l)  # record matrix; each row is a weight vector for the iteration
        self.weights_r = np.random.rand(1, self.lgn_r, )  # record matrix; each row is a weight vector for the iteration
        self.theta = 0.15 * np.random.rand()  # sliding threshold initialized to a number btw 0 and 0.15
        self.thresholds = self.theta  # record array of thresholds
        self.tuning_curve_l = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration
        self.tuning_curve_r = np.zeros((self.patterns,))  # record matrix; each row is a tuning curve for the iteration

    def train(self, da_l, da_r, ds_l=0, ds_r=0, nl=0, nr=0):
        # da = actual pattern; ds = spontaneous; n = non-lgn noise

        ca = sum(self.ml * da_l[0]) + sum(self.mr * da_r[0])
        cs = sum(self.ml * ds_l[0]) + sum(self.mr * ds_r[0])
        cn = sum(self.ml * nl[0]) + sum(self.mr * nr[0])

        if (self.lgn_l != np.shape(da_l)[1]) or (self.lgn_r != np.shape(da_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        elif (self.lgn_l != np.shape(ds_l)[1]) or (self.lgn_r != np.shape(ds_r)[1]):
            raise Exception('pattern size must equal weight (i.e lgn) size ')
        else:
            for i in range(self.iterations):
                t = i * self.eta
                k = np.random.randint(np.shape(da_l)[0])
                # print(da_r[k])

                ca = np.sum(self.ml * da_l[k]) + np.sum(self.mr * da_r[k])
                cs = np.sum(self.ml * ds_l[k]) + np.sum(self.mr * ds_r[k])
                cn = np.sum(self.ml * nl[k]) + np.sum(self.mr * nr[k])
                c = ca + cs + cn
                phi = c * (c - self.theta)

                self.ml = self.ml + self.eta * phi * (da_l[k] + ds_l[k] + nl[k])  # update left weights
                # print(self.ml)
                self.mr = self.mr + self.eta * phi * (da_r[k] + ds_r[k] + nr[k])  # update right weights
                # print(self.ml)
                # Add new weights to the weight matrices
                self.weights_l = np.vstack((self.weights_l, self.ml))
                self.weights_r = np.vstack((self.weights_r, self.mr))
                # We need to now update theta
                ca_bar = (1 / self.tau) * quad(lambda s: ca * np.exp(-(t - s) / self.tau), -np.inf, t)[0]
                self.theta = (ca_bar ** self.p) / self.c0

                # self.theta = self.theta + (1/self.tau)*(ca**self.p-self.theta)
                print(self.theta)
                # Add new theta to the vector of thresholds
                self.thresholds = np.hstack((self.thresholds, self.theta))
                # Add responses to all orientation in the tuning curve matrix
                self.tuning_curve_l = np.vstack((self.tuning_curve_l, np.sum((da_l + ds_l + nl) * self.ml, axis=1)))
                self.tuning_curve_r = np.vstack((self.tuning_curve_r, np.sum((da_r + ds_r + nr) * self.mr, axis=1)))
        # Now delete the first entries (initializations) of the record matrices/vectors
        self.weights_l = np.delete(self.weights_l, 0, axis=0)
        self.weights_r = np.delete(self.weights_r, 0, axis=0)
        self.thresholds = np.delete(self.thresholds, 0)
        self.tuning_curve_l = np.delete(self.tuning_curve_l, 0, axis=0)
        self.tuning_curve_r = np.delete(self.tuning_curve_r, 0, axis=0)

        return self.weights_l, self.weights_r, self.thresholds, self.tuning_curve_l, self.tuning_curve_r

    # def train_nr(self):

    # def train_md(self):

    # def train_rs(self):

    # def train_bd(self):

    # def train_re(self):

    # def train_st(self):

    # def get_response(self):

    def plot_weights(self):
        (row_l, col_l) = np.shape(self.weights_l)
        print(row_l)
        print(col_l)
        (row_r, col_r) = np.shape(self.weights_r)
        print(row_r)
        print(col_r)
        x = list(range(0, self.iterations))
        x = self.eta * np.array(x)
        figure, axis = plt.subplots(self.lgn_l, 2)

        for k in range(col_l):
            axis[k, 0].plot(x, self.weights_l[:, k])
            axis[k, 0].set_xlabel("Time")
            axis[k, 0].set_ylabel("Weight")
            axis[k, 0].set_title(" Time vs Weight")

        for k in range(col_r):
            axis[k, 1].plot(x, self.weights_r[:, k])
            axis[k, 1].set_xlabel("Time")
            axis[k, 1].set_ylabel("Weight")
            axis[k, 1].set_title(" Time vs Weight")

        plt.legend()
        plt.show()


def main():
    pat = 0.001 * np.random.rand(3, 3) + np.identity(3)
    print(pat)
    neuron = BCMNeuron(lgn_l=3, lgn_r=3, patterns=3)
    wl, ws, th, tcl, tcr = neuron.train(da_l=pat, da_r=pat, ds_l=0 * pat, ds_r=0 * pat, nl=0 * pat, nr=0 * pat)
    neuron.plot_weights()

if __name__ == "__main__":
    main()