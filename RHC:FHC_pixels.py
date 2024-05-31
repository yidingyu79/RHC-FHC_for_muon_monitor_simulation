#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:04:12 2024

@author: yidingyu
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import inspect
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
from scipy.stats import pearsonr
import scipy.stats
import pandas as pd
import math
import seaborn as sns

# Set the default figure size and font size for all plots
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12  # Adjust this value as needed

# Load the first .npz file
npzfile1 = np.load('/Users/yidingyu/npz/Mag_study/nominal_x0.1_y-0.3.npz')

# Extract data from the .npz file
mm1x0mm = npzfile1['z5x0mm']
mm2x0mm = npzfile1['z6x0mm']
mm3x0mm = npzfile1['z7x0mm']
mm1y0mm = npzfile1['z5y0mm']
mm2y0mm = npzfile1['z6y0mm']
mm3y0mm = npzfile1['z7y0mm']

mm1nimpwt0mm = npzfile1["nimpwtz5"]
mm2nimpwt0mm = npzfile1["nimpwtz6"]
mm3nimpwt0mm = npzfile1["nimpwtz7"]

# Create 2D histograms for the first dataset
mm1xy_0mm, xbins0mm, ybins0mm, im0mm = plt.hist2d(mm1x0mm * -1, mm1y0mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm1nimpwt0mm)
mm2xy_0mm, xbins0mm, ybins0mm, im0mm = plt.hist2d(mm2x0mm * -1, mm2y0mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm2nimpwt0mm)
mm3xy_0mm, xbins0mm, ybins0mm, im0mm = plt.hist2d(mm3x0mm * -1, mm3y0mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm3nimpwt0mm)

# Load the second .npz file (commented out lines are different file options)
npzfile2 = np.load('/Users/yidingyu/npz/Mag_study/RHC_x0.1_y-0.3.npz')

# Extract data from the second .npz file
mm1x1mm = npzfile2['z5x0mm']
mm2x1mm = npzfile2['z6x0mm']
mm3x1mm = npzfile2['z7x0mm']
mm1y1mm = npzfile2['z5y0mm']
mm2y1mm = npzfile2['z6y0mm']
mm3y1mm = npzfile2['z7y0mm']

mm1nimpwt1mm = npzfile2["nimpwtz5"]
mm2nimpwt1mm = npzfile2["nimpwtz6"]
mm3nimpwt1mm = npzfile2["nimpwtz7"]

# Create 2D histograms for the second dataset
mm1xy_1mm, xbins1mm, ybins1mm, im1mm = plt.hist2d(mm1x1mm * -1, mm1y1mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm1nimpwt1mm)
mm2xy_1mm, xbins1mm, ybins1mm, im1mm = plt.hist2d(mm2x1mm * -1, mm2y1mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm2nimpwt1mm)
mm3xy_1mm, xbins1mm, ybins1mm, im1mm = plt.hist2d(mm3x1mm * -1, mm3y1mm, [9, 9], [[-1142, 1142], [-1142, 1142]], weights=mm3nimpwt1mm)

# Define a function to get the number of events and their errors
def getevents(h1, a, b):
    events = h1[a, b]
    events_err = np.sqrt(h1[a, b])
    return events, events_err

# Define a function to compare two histograms and calculate the ratio and its error
def compare(h1, h2):
    ratio_m = np.zeros((9, 9))
    ratio_m_err = np.zeros((9, 9))
    for a in range(9):
        for b in range(9):
            u_n, u_err = getevents(h1, a, b)
            n_n, n_err = getevents(h2, a, b)
            ratio = u_n / n_n
            ratio_err = np.sqrt(u_err * u_err + n_err * n_err) / n_n
            ratio_m[a, b] = ratio
            ratio_m_err[a, b] = ratio_err
    return ratio_m, ratio_m_err

# Define a function to compare two histograms and calculate the normalized ratio and its error
def compare_err(h1, h2):
    ratio_m = np.zeros((9, 9))
    ratio_m_err = np.zeros((9, 9))
    for a in range(9):
        for b in range(9):
            u_n, u_err = getevents(h1, a, b)
            n_n, n_err = getevents(h2, a, b)
            ratio = u_n / n_n
            ratio_err = np.sqrt(u_err * u_err + n_err * n_err) / n_n
            ratio_m[a, b] = (ratio - 1) / ratio_err
            ratio_m_err[a, b] = ratio_err
    return ratio_m, ratio_m_err

# Get the number of events and their errors for a specific bin in the first histogram
sim, sim_err = getevents(mm1xy_0mm, 4, 4)
print(sim)

# Normalize the first histogram by dividing by 100000
mm1xy_sim = mm1xy_0mm / 100000

# Rotate the normalized histogram by 90 degrees counter-clockwise
mm1xy_sim_2 = np.zeros((9, 9))
for a in range(9):
    for b in range(9):
        mm1xy_sim_2[a, b] = mm1xy_sim[b, 8 - a]

# Plot the first normalized and rotated histogram
plt.figure(2)
plt.title(' FHC ( mm1 ) ')
ax = sns.heatmap(mm1xy_sim_2, cmap='coolwarm', linewidths=0.2, annot=True, fmt='.3g', vmin=0.0, vmax=8.0,
                 xticklabels=[-1015, -753, -507, -253, 0, 253, 507, 753, 1015], yticklabels=[1015, 753, 507, 253, 0, -253, -507, -753, -1015])
ax.set_xlabel('Horizontal position (mm)')
ax.set_ylabel('Vertical position (mm)')

# Get the number of events and their errors for a specific bin in the second histogram
sim, sim_err = getevents(mm1xy_1mm, 4, 4)
print(sim)

# Normalize the second histogram by dividing by 100000
mm1xy_sim = mm1xy_1mm / 100000

# Rotate the normalized histogram by 90 degrees counter-clockwise
mm1xy_sim_2_1mm = np.zeros((9, 9))
for a in range(9):
    for b in range(9):
        mm1xy_sim_2_1mm[a, b] = mm1xy_sim[b, 8 - a]

# Plot the second normalized and rotated histogram
plt.figure(3)
plt.title(' RHC ( mm1 ) ')
ax = sns.heatmap(mm1xy_sim_2_1mm, cmap='coolwarm', linewidths=0.2, annot=True, fmt='.3g', vmin=0.0, vmax=8.0,
                 xticklabels=[-1015, -753, -507, -253, 0, 253, 507, 753, 1015], yticklabels=[1015, 753, 507, 253, 0, -253, -507, -753, -1015])
ax.set_xlabel('Horizontal position (mm)')
ax.set_ylabel('Vertical position (mm)')

# Compare the two normalized histograms and calculate the ratio and its error
mm1_m, mm1_m_err = compare(mm1xy_sim_2_1mm, mm1xy_sim_2)

# Plot the ratio of the two histograms
plt.figure(1)
plt.title('RHC/FHC ( mm1 ) ')
ax = sns.heatmap(mm1_m, cmap='coolwarm', linewidths=0.2, annot=True, fmt='.3g',
                 xticklabels=[-1015, -753, -507, -253, 0, 253, 507, 753, 1015], yticklabels=[1015, 753, 507, 253, 0, -253, -507, -753, -1015])
ax.set_xlabel('Horizontal position (mm)')
ax.set_ylabel('Vertical position (mm)')

