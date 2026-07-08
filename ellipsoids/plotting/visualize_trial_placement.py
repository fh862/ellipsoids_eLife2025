#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 11:47:02 2025

@author: fangfang

The purpose of this script is solely to illustrate trial placement.
While other scripts (e.g., sim_4d6d_color_discrimination.py, fit_4d_human.py)
also visualize trial placement, they include additional components such as model fitting.
In contrast, this script is focused exclusively on visualizing trial placement, 
with no other functionality.

"""

import matplotlib.pyplot as plt
import dill as pickled
from dataclasses import replace
import re
import numpy as np
import os
from plotting.adaptive_sampling_plotting import SamplingRefCompPairVisualization, Plot2DSamplingSettings
from plotting.wishart_plotting import PlotSettingsBase 
from analysis.utils_load import select_file_and_get_path, extract_sub_number

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
"""
We have 4 options:
    1. Experimental adaptive trials (Adaptively sampled AEPsych trials)
    2. Simulated adaptive trials (Adaptively sampled AEPsych trials based on a known ground truth)
    3. Experimental MOCS trials
    4. Simulated MOCS trials (based on a known ground truth)
    
Case 1: ELPS_analysis/Experiment_DataFiles/sub#/fits
    'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub#_decayRate0.5_nBasisDeg5.pkl'

Case 2: META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub#/decayRate0.5
    'Fitted_byWishart_Isoluminant plane_4DExpt_300_300_300_5100_AEPsychSampling_EAVC_decayRate0.5_nBasisDeg5_sub#.pkl'
    
Case 3: META_analysis/ModelFitting_DataFiles/4dTask/CIE/sub#/decayRate0.5
    'Fitted_weibull_psychometric_func_Isoluminant plane_6000totalTrials_25refs_MOCS_sub#.pkl'

Case 4: ELPS_analysis/Simulation_DataFiles/Isoluminant plane/MOCS/gt_CIE
    'Sim2dTask_colorDiscrimination_Isoluminant plane_MOCStrials_25refs_12levels_20trialsPerLevel_subCIE1994_Sobol_seed2000.pkl'
"""
input_fileDir_fits, file_name = select_file_and_get_path()

# Full path to the selected fitted-data pickle.
full_path = os.path.join(input_fileDir_fits, file_name)

# Save trial-placement figures beside the fit folder, under the matching FigFiles tree.
output_figDir_fits = re.sub(
    r'DataFiles', 'FigFiles',
    os.path.join(os.path.dirname(input_fileDir_fits), 'trial_placement')
)

# Create the output folder on demand.
os.makedirs(output_figDir_fits, exist_ok=True)

#%% 
# ---------------------------------------------------------------------------
# SECTION 2: Load the necessary variables from the file
# --------------------------------------------------------------------------
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)
# Color-space metadata and transforms used by the plotting helper.
color_thres_data = vars_dict['color_thres_data']

# MOCS trials are optional; fitted AEPsych pickles may include them as validation
flag_has_MOCS = vars_dict.get('xref_MOCS_list') is not None

if flag_has_MOCS:
    try: #experimental MOCS data
        subN = extract_sub_number(file_name)
        xref_MOCS = np.concatenate(vars_dict['refStimulus_MOCS'])
        x1_MOCS = np.concatenate(vars_dict['compStimulus_MOCS'])
        
    except KeyError: #simulated MOCS data
        subN = re.search(r'(?<=sub)([^_]+)', file_name)[0]
        nUnique_cond = vars_dict['nRefs']*vars_dict['nLevels']
        xref_MOCS = vars_dict['MOCS_xref_shuffled'][:nUnique_cond]
        x1_MOCS = vars_dict['MOCS_x1_shuffled'][:nUnique_cond]    

# AEPsych trials are always plotted; simulated files use a different saved object name.
subN = extract_sub_number(file_name)
try: #experimental AEPsych data
    expt_trial = vars_dict['expt_trial']
    nTrials_strat = vars_dict.get("nTrials_strat", vars_dict.get("NTRIALS_STRAT"))   
    
    # Use the realized AEPsych count; configured capacity can exceed the trials run.
    # `expt_trial` may also contain inserted pregenerated Sobol trials, so count
    # the AEPsych-only stream saved by the loader instead.
    nTrials_strat[-1] = np.concatenate(vars_dict['aepsych_data'][0]).shape[0] - \
        np.sum(nTrials_strat[:-1])
    
except KeyError: #simulated AEPsych data
    expt_trial = vars_dict['AEPsych_trial_given_Wishart_gt']
    strat_dict = vars_dict['strat_dict']
    nTrials_strat = list(vars_dict['strat_dict'].values())
    
nTrials_AEPsych = sum(nTrials_strat)
xref = expt_trial.xref_all
x1 = expt_trial.x1_all
nTrials_total = xref.shape[0]

#%%
# -----------------------------------------------------------------------
# SECTION 3: Visualize trial placement (separately for Sobol and EAVC)
# -----------------------------------------------------------------------
# Shared plotting settings; per-figure fields are replaced inside the loop.
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)
pltSettings_tp = replace(Plot2DSamplingSettings(), **pltSettings_base.__dict__)
sampling_vis = SamplingRefCompPairVisualization(2, color_thres_data,
                                                settings = pltSettings_tp,
                                                save_fig = False)

# AEPsych slices: initial Sobol, adaptive-only, inserted pregenerated Sobol, and all trials.
marker_alpha = [0.3, 0.2, 0.2, 0.1]
slc_datapoints_to_show_lb = [0,                       
                             sum(nTrials_strat[:-1]),   
                             nTrials_AEPsych,             
                             0]
slc_datapoints_to_show_ub = [sum(nTrials_strat[:-1]),         
                             nTrials_AEPsych,     
                             nTrials_total, 
                             nTrials_total]
plot_xref = [xref]
plot_x1 = [x1]
plot_marker_alpha = [marker_alpha]
plot_slc_lb = [slc_datapoints_to_show_lb]
plot_slc_ub = [slc_datapoints_to_show_ub]
plot_str_ext = ['']

if flag_has_MOCS:
    # Append MOCS as a second dataset so the same plotting loop handles both streams.
    marker_alpha_MOCS = [0.6]
    slc_datapoints_to_show_lb_MOCS = [0]
    slc_datapoints_to_show_ub_MOCS = [xref_MOCS.shape[0]]
    plot_xref.append(xref_MOCS)
    plot_x1.append(x1_MOCS)
    plot_marker_alpha.append(marker_alpha_MOCS)
    plot_slc_lb.append(slc_datapoints_to_show_lb_MOCS)
    plot_slc_ub.append(slc_datapoints_to_show_ub_MOCS)
    plot_str_ext.append('_MOCS')

# Plot AEPsych first, then MOCS if present.
for xref_plot, x1_plot, marker_alpha_plot, slc_lb_plot, slc_ub_plot, str_ext in zip(
        plot_xref, plot_x1, plot_marker_alpha, plot_slc_lb, plot_slc_ub, plot_str_ext):
    for i, (lb_i, ub_i) in enumerate(zip(slc_lb_plot, slc_ub_plot)):
        str_idx = f'{ub_i:05}total' if lb_i == 0 else f'{ub_i:05}total_from{lb_i:05}'
        fig_name = f"TrialPlacement{str_ext}_isothreshold_{color_thres_data.plane_2D}_{str_idx}_sub{subN}"
        pltSettings_tp = replace(pltSettings_tp,
                                 ref_markeralpha = 0.6,#marker_alpha_plot[i],
                                 comp_markeralpha = marker_alpha_plot[i],
                                 linealpha = marker_alpha_plot[i], 
                                 ticks = np.linspace(-0.7, 0.7, 5),
                                 bounds = 0.75 * np.array([-1,1]),
                                 fig_name = fig_name)

        fig, ax = plt.subplots(1, 1, figsize = (3,3.5), dpi= pltSettings_tp.dpi)
        sampling_vis.plot_sampling(xref_plot[lb_i:ub_i],
                                   x1_plot[lb_i:ub_i],
                                   settings = pltSettings_tp,
                                   ax = ax)
        ax.set_title(f'{color_thres_data.plane_2D}')
        fig.savefig(os.path.join(output_figDir_fits, f'{fig_name}.pdf'), bbox_inches='tight')    
        plt.show()
