#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:32:26 2025

@author: fangfang
"""

import matplotlib.pyplot as plt
import dill as pickled
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.append("/Users/fangfang/Documents/MATLAB/projects/ellipsoids/ellipsoids")
from analysis.color_thres import color_thresholds
from analysis.MOCS_thresholds import sim_MOCS_trials

#--------------------------------------------------------------------------
# SECTION 1: Pre-generate Sobol Trials
#--------------------------------------------------------------------------
# We want to slot in pre-generated Sobol trials when AEPsych is taking longer
# than expected for generating the next trial. Since we don’t know exactly how 
# many Sobol trials will be needed, we pregenerate more than necessary, but 
# most will not be used in the actual experiment.

# 6D psychometric field
stim_dims = 2
psyfield_dims = 4  #6 or 4

# base directory
baseDir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

# this class contains useful transformations
color_thres_data = color_thresholds(stim_dims, baseDir, 
                                    plane_2D= 'Isoluminant plane' if stim_dims == 2 else None)
                                    #LSisolating plane
if stim_dims == 2:
    #eLife paper: "02242025"
    #adaptation: "10062025"
    #dichromat: "11172025"
    #adaptation (round 2): "02012026"
    color_thres_data.load_transformation_matrix(file_date="02012026") 
    
# Number of Sobol trials per session
nTrials_sobol_perSession = 900   #should be multiples of 3 (the number of sobol_scaler)

# specify a unique seed
sobol_seed = 1770

# Lower and upper bounds for the 4D Sobol samples (representing different dimensions)
#3D:
#lb_sobol_trials = [-0.85, -0.85, -0.85, -0.15, -0.15, -0.15]  
#ub_sobol_trials = [ 0.85,  0.85,  0.85,  0.15,  0.15,  0.15]  
#2D
lb_sobol_trials = [-0.75, -0.75, -0.25, -0.25]  
ub_sobol_trials = [ 0.75,  0.75,  0.25,  0.25]  
#2D (dichromat updated)
# lb_sobol_trials = [-0.55, -0.7, -0.45, -0.3]  
# ub_sobol_trials = [ 0.55,  0.7,  0.45,  0.3]  

# Scaling factors applied to the comparison stimulus to balance trial difficulty
sobol_scaler = [2/8, 3/8, 4/8]
#2D (dichromat updated)
#sobol_scaler = [4/8, 6/8, 1]

if nTrials_sobol_perSession % len(sobol_scaler) != 0:
    raise ValueError('The pregenerated trials has to be multiples of the number of sobol scalers!')

# Number of times to repeat the scaling factor set to match the number of trials
num_repeats = nTrials_sobol_perSession // len(sobol_scaler)

# Maximum number of experimental sessions (we generate more than needed)
nSessions = 60

# Preallocate arrays to store generated Sobol reference (`xref`) and comparison (`x1`) stimuli
Sobol_xref = np.full((nSessions, nTrials_sobol_perSession, psyfield_dims // 2), np.nan)
Sobol_x1   = np.full(Sobol_xref.shape, np.nan)

flag_debugplots = True #whether we would like to visualize the pregenerated trials
flag_addCatchTrials = True
if flag_addCatchTrials:
    percent_catchTriasl = 0.05
    delta_catchTrials_unique = np.array([[-0.25, -0.25],
                                         [-0.25, 0.25],
                                         [0.25, -0.25],
                                         [0.25, 0.25]])
    # updated for the dichromat
    # delta_catchTrials_unique = np.array([[-0.45, -0.3],
    #                                      [-0.45, 0.3],
    #                                      [0.45, -0.3],
    #                                      [0.45, 0.3]])
    nTotal_catchTrials = int(nTrials_sobol_perSession * percent_catchTriasl)
    #initialize
    catch_idx_all = np.full((nSessions, nTotal_catchTrials), np.nan)
    choice_unique_catch_all = np.full(catch_idx_all.shape, np.nan)
    delta_catch_all = np.full((nSessions, nTotal_catchTrials, stim_dims), np.nan)
    
for n in range(nSessions):
    # One RNG per session for reproducibility and isolation
    rng = np.random.default_rng(sobol_seed + n)

    # Sobol samples
    Sobol_samples = sim_MOCS_trials.sample_sobol(
        nTrials_sobol_perSession, lb=lb_sobol_trials, ub=ub_sobol_trials,
        force_center=False, seed=sobol_seed + n
    )

    # Shuffle scalers reproducibly
    sobol_scaler_n = np.concatenate([rng.permutation(sobol_scaler) for _ in range(num_repeats)])

    # Assign xref/x1
    Sobol_xref[n] = Sobol_samples[:, :stim_dims]
    Sobol_x1[n]   = Sobol_xref[n] + sobol_scaler_n[:, None] * Sobol_samples[:, stim_dims:]

    # --- Catch trials (fixed) ---
    if flag_addCatchTrials and nTotal_catchTrials > 0:
        catch_idx = rng.choice(nTrials_sobol_perSession, size=nTotal_catchTrials, replace=False)
        catch_idx_all[n] = catch_idx
        # Sample deltas with replacement to get exactly nTotal_catchTrials rows
        choose = rng.integers(0, len(delta_catchTrials_unique), size=nTotal_catchTrials)
        choice_unique_catch_all[n] = choose
        
        #delta values for the catch trials
        delta_catch = delta_catchTrials_unique[choose]
        delta_catch_all[n] = delta_catch
        
        #rewrite the comparison stimuli
        Sobol_x1[n, catch_idx] = Sobol_xref[n, catch_idx] + delta_catch
    
    if flag_debugplots:
        if stim_dims == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot reference points
            #ax.scatter(Sobol_xref[n, :, 0], Sobol_xref[n, :, 1], Sobol_xref[n, :, 2], 
            #    marker='.', c = color_thres_data.W_unit_to_N_unit(Sobol_xref[n, m]))
            
            # Plot comparison points
            #ax.scatter(Sobol_x1[n, :, 0], Sobol_x1[n, :, 1], Sobol_x1[n, :, 2],
            #    marker='o', s=1, c = color_thres_data.W_unit_to_N_unit(Sobol_x1[n,m]))
            
            #Draw lines between reference and comparison points
            for m in range(nTrials_sobol_perSession):
                ax.plot([Sobol_xref[n, m, 0], Sobol_x1[n, m, 0]],
                        [Sobol_xref[n, m, 1], Sobol_x1[n, m, 1]],
                        [Sobol_xref[n, m, 2], Sobol_x1[n, m, 2]], 
                        c = color_thres_data.W_unit_to_N_unit(Sobol_xref[n, m]), alpha = 0.5)
            
            plt.show()
        elif stim_dims == 2:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
            # Per-trial lines
            for m in range(nTrials_sobol_perSession):
                rgb = color_thres_data.W2D_to_rgb(Sobol_xref[n, m])
                rgb = np.clip(rgb, 0, 1)  # keep valid for Matplotlib
                ax.plot([Sobol_xref[n, m, 0], Sobol_x1[n, m, 0]],
                        [Sobol_xref[n, m, 1], Sobol_x1[n, m, 1]],
                        c=rgb, alpha=0.5, linewidth=0.6)

            # Draw catch trials once, outside the loop
            if flag_addCatchTrials and nTotal_catchTrials > 0:
                ax.plot([Sobol_xref[n, catch_idx, 0], Sobol_x1[n, catch_idx, 0]],
                        [Sobol_xref[n, catch_idx, 1], Sobol_x1[n, catch_idx, 1]],
                        c='k', alpha=0.5, linewidth=0.8)
            ax.set_xlim([-1,1]); ax.set_ylim([-1,1])
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.show()

#%%
#-------------------------------------------------------------------------- 
# SECTION 2: Save the data
#--------------------------------------------------------------------------
cspace = 'RGBcube' if stim_dims == 3 else color_thres_data.plane_2D.replace(" ", "_")
output_file = f'Sim{psyfield_dims}dTask_colorDiscrimination_{cspace}_'+\
                f'pregeneratedSobol_seed{sobol_seed}.pkl'
output_fileDir = os.path.join(baseDir, 'ELPS_analysis', 'Simulation_DataFiles',
                              f'{stim_dims}D', 'pregenerated_Sobol')
os.makedirs(output_fileDir, exist_ok= True)
full_path2 = os.path.join(output_fileDir, output_file)

variable_names = ['nTrials_sobol_perSession', 'lb_sobol_trials','ub_sobol_trials',
                  'sobol_scaler', 'Sobol_xref', 'Sobol_x1', 'catch_idx_all', 
                  'choice_unique_catch_all', 'delta_catch_all']
vars_dict = {}

for var_name in variable_names:
    try:
        # Check if the variable exists in the global scope
        vars_dict[var_name] = eval(var_name)
    except NameError:
        # If the variable does not exist, assign None and print a message
        vars_dict[var_name] = None
        print(f"Variable '{var_name}' does not exist. Assigned as None.")

# Write the list of dictionaries to a file using pickle
with open(full_path2, 'wb') as f:
    pickled.dump(vars_dict, f)