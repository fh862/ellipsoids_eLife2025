#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:10:00 2025

@author: fangfang

This script visualizes the best-fitting Wishart-process weights as a function
of the order of the 2D Chebyshev polynomial basis functions.

Specifically, it:
    • Loads a fitted WPPM model from disk
    • Extracts the estimated weight matrix W
    • Computes the polynomial basis orders (i + j)
    • Overlays the learned weights with the corresponding prior variance
      (±2√η) implied by the chosen hyperparameters
"""

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import dill as pickled
from dataclasses import replace
import matplotlib.pyplot as plt
import os
from plotting.wishart_plotting import (
    WishartModelBasicsVisualization,
    PlotSettingsBase,
    PlotBasis1DSettings,
    PlotWAllSettings
)
from analysis.utils_load import select_file_and_get_path

#%% 
# ---------------------------------------------------------------------
# Load fitted WPPM results
# ---------------------------------------------------------------------
# Select the pickle file containing the fitted model
# Example path:
#   'ELPS_analysis/Experiment_DataFiles/pilot2/sub11/fits/'
# Example filename:
#   'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub11_... .pkl'
input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Define output directory for figures (mirrors DataFiles → FigFiles)
output_figDir = os.path.join(input_fileDir_fits.replace('DataFiles', 'FigFiles'))
os.makedirs(output_figDir, exist_ok=True)

# Base plotting settings shared across figures
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir)

# Specialized settings for 1D basis plots
pltSettings_1D = replace(PlotBasis1DSettings(), **pltSettings_base.__dict__)

# Visualization helper (figure saving handled manually below)
visualize_basis = WishartModelBasicsVisualization(
    save_fig=False,
    settings=pltSettings_base
)

#%% 
# ---------------------------------------------------------------------
# Load model and extract fitted parameters
# ---------------------------------------------------------------------
with open(full_path, 'rb') as f:
    data_load = pickled.load(f)

# Extract fitted weight matrix and model object
W_est = data_load['model_pred_Wishart'].W_est
model = data_load['model_pred_Wishart'].model

# Total Chebyshev order at each tensor-product basis location
basis_orders = jnp.indices(
    (model.degree,) * model.num_dims
).sum(axis=0)

# Match the complete shape of W_est
basis_orders_rep = jnp.broadcast_to(
    basis_orders[..., None, None],
    W_est.shape
)

# Compute prior variance envelope (±2√η)
# Extract prior hyperparameters
gamma = model.variance_scale
eps_all = [0.4]  # decay rate(s) to visualize,, we can keep adding more

# Unique basis orders
d = np.unique(basis_orders_rep)

# Compute prior variance for each basis order
eta_all = [gamma * e ** d for e in eps_all]

# ±2 standard deviation envelope
eta_sqrt = 2 * np.sqrt(eta_all)

#%%
# --------------------------------------------------------------------
# Plot best-fitting weights and prior envelope
# ---------------------------------------------------------------------
# Plot settings for weight visualization
pltSettings_W_all = replace(PlotWAllSettings(), **pltSettings_base.__dict__)
pltSettings_W_all = replace(
    pltSettings_W_all,
    ybds=[-0.045, 0.045], #[-0.06, 0.06],
    yticks= np.linspace(-0.04, 0.04, 5),#np.linspace(-0.06, 0.06, 5),
    marker_alpha=0.4,
    ylabel='Best-fitting weights',
    xlabel= f'The order of {model.num_dims}D Chebyshev basis functions '
    f'$({" + ".join("ijk"[:model.num_dims])})$'
)

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize= (8, 3.5), dpi=pltSettings_W_all.dpi)

# Line styles and colors for prior envelope
ls = ['-', '-.', ':']  # we can keep adding more
cmap = ['k', 'g', 'r'] # we can keep adding more

# Plot ±2√η prior envelope
for n in range(len(eps_all)):
    ax.plot(d, eta_sqrt[n], color=cmap[n], ls=ls[n], lw=0.75,
        label=fr'$\pm 2\sqrt{{\eta}}$ at $\epsilon = {eps_all[n]:.1f}$, $\gamma = {gamma:.4f}$')
    ax.plot( d, -eta_sqrt[n], color=cmap[n], ls=ls[n], lw=0.75)
ax.legend(loc='lower right', fontsize=10)

# Overlay best-fitting weights
visualize_basis.plot_W_all(W_est, basis_orders_rep, settings=pltSettings_W_all, ax=ax)

# Save figure
fig.savefig(os.path.join(output_figDir, f"Bestfit_W_{file_name[:-4]}.pdf"),
    bbox_inches='tight')