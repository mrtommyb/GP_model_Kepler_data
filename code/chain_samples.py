#!/usr/bin/env python
# -*- coding: utf-8 -*-
## turn koi2133 chains into data model

from __future__ import division, print_function

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import george
from george.kernels import ExpSquaredKernel, RBFKernel, ExpKernel

from ktransit import LCModel, FitTransit

from scipy import optimize

import h5py

def get_sample(chain_vals):
    M = LCModel()
    M.add_star(rho=mle[0],zpt=mle[1],ld1=mle[2],
        ld2=mle[3],veloffset=mle[4])
    M.add_planet(T0=mle[7],period=mle[8],impact=mle[9],
        rprs=mle[10],ecosw=mle[11],esinw=mle[12],
        rvamp=mle[13],occ=mle[14],ell=mle[15],alb=mle[16])
    M.add_data(time=f['time'][:])
    M.add_rv(rvtime=f['rvtime'])

    #kernel = ((mle[5]**2 * RBFKernel(mle[6])) +
    #        (mle[7]**2 * RBFKernel(mle[8])))
    #kernel = ((mle[5]**2 * ExpKernel(mle[6])) +
    #        (mle[7]**2 * RBFKernel(mle[8])))
    kernel = mle[5]**2 * RBFKernel(mle[6])
    gp = george.GaussianProcess(kernel)

    sample = np.array([])
    for i in np.arange(len(f['time'][:]) // 1000):
        section = np.arange(i*1000,i*1000 + 1000)
        gp.compute(f['time'][:][section], f['err'][:][section])
        sample = np.r_[sample,gp.predict(
            f['flux'][:][section] - M.transitmodel[section],f['time'][:][section])[0]]
    return sample, M.transitmodel


if __name__ == '__main__':
    fn = 'koi2133_np1_priorTrue_dil0.0GP.hdf5'
    f = h5py.File(fn)
    g = f['mcmc']['chain'][:]
    lnprob = f['mcmc']['lnprob'][:]

    mle_idx = np.unravel_index(lnprob.argmax(),
            lnprob.shape)

    mle = g[mle_idx]

    sample, tmod = get_sample(mle)








