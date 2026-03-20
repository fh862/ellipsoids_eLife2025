#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 12:01:37 2025

@author: fangfang

The goal of this script is to load the Wishart model predictions to the 3D / 6D 
data. As of Sep 17th, the only dataset is just my own pilot data. The script 
does the following:
    1. visualize threshold ellipsoids in the RGB cube
    

"""

import jax
jax.config.update("jax_enable_x64", True)
import dill as pickled
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from dataclasses import replace
from scipy.io import loadmat
from skimage.measure import EllipseModel
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys
import os
script_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from analysis.utils_load import select_file_and_get_path
from analysis.conf_interval import find_inner_outer_contours_for_gridRefs
from analysis.ellipsoids_tools import slice_ellipsoid_byPlane
from plotting.wishart_predictions_plotting import WishartPredictionsVisualization
from plotting.wishart_plotting import PlotSettingsBase 
from plotting.wishart_predictions_plotting import Plot2DPredSettings
from plotting.sim_CIELab_plotting import CIELabVisualization, Plot3DSettings

base_dir = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/'

#%%
#---------------------------------------------------------------------------
# SECTION 1: load the model fits to the empirical data
# --------------------------------------------------------------------------
# Select the file containing the model fits
# Navigate to the directory: ELPS_analysis/Experiment_DataFiles/pilot3/sub1/fits
# 'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5.pkl'

input_fileDir_fits, file_name = select_file_and_get_path()

# Construct the full path to the selected file
full_path = os.path.join(input_fileDir_fits, file_name)

# Load the necessary variables from the file
with open(full_path, 'rb') as f:
    vars_dict = pickled.load(f)

# - Transformation matrices for converting between DKL, RGB, and W spaces
color_thres_data = vars_dict['color_thres_data']
color_thres_data.base_path = base_dir
color_thres_data.load_transformation_matrix()

# - Dimensionality of the color space (e.g., 2D for isoluminant planes)
ndims = color_thres_data.color_dimension

# Create the output directory if it doesn't exist
output_figDir_fits = input_fileDir_fits.replace('DataFiles', 'FigFiles')
os.makedirs(output_figDir_fits, exist_ok=True)

#%% load variables
model_pred_3D = vars_dict['model_pred_Wishart']
grid_3D = vars_dict['grid']
grid_3D_trans = vars_dict['grid_trans']
num_grid_pts_3D = vars_dict['NUM_GRID_PTS']

model_pred = vars_dict['model_pred_Wishart_grid_isoluminant']
model = model_pred.model
num_grid_pts = vars_dict['NUM_GRID_PTS_2D']
grid = vars_dict['grid_isoluminant_3DW']
grid_trans = vars_dict['grid_isoluminant_3DW_trans']

#%%
# Create an instance of the class
pltSettings_base = PlotSettingsBase(fig_dir= output_figDir_fits, fontsize = 11)
Plot3D_settings = replace(Plot3DSettings(), **pltSettings_base.__dict__)
Plot3D_settings = replace(Plot3D_settings,
                          fig_size = (6,6),
                          visualize_thresholdPoints = False,
                          visualize_ellipsoids = True,
                          lim = [-1,1],
                          ticks = np.linspace(-0.7, 0.7,5),
                          surf_alpha = 0.5,
                          flag_input_W = True,
                          title = None,
                          view_angle = [30,-120])

vis = CIELabVisualization(color_thres_data, settings = pltSettings_base)

#create a figure
fig1 = plt.figure(figsize=Plot3D_settings.fig_size, dpi=Plot3D_settings.dpi,
                  constrained_layout=True)
ax1 = fig1.add_subplot(111, projection='3d')
vis.plot_3D(np.reshape(grid_3D_trans,(num_grid_pts_3D**ndims, ndims)),
            np.reshape(model_pred_3D.fitEll_scaled,(num_grid_pts_3D** ndims, ndims,-1)),
            ax = ax1,
            settings = Plot3D_settings)
ax1.legend(bbox_to_anchor=(0.5, -0.2), fontsize= Plot3D_settings.fontsize - 1)
# Save the figure as a PDF
#fig1.savefig(os.path.join(output_figDir_fits, f"Ellipsoids_{file_name[:-4]}.pdf"))    
plt.show()

#%%
path_str = "/Users/fangfang/Documents/MATLAB/projects/ColorEllipsoids/FilesFromPsychtoolbox/"
os.chdir(path_str)
#load data
#iso_mat = loadmat('W_from_PlanarGamut.mat')
mat_file = loadmat('Transformation_btw_color_spaces.mat')
iso_mat = mat_file['DELL_02242025_texture_right'][0]
gamut_rgb = color_thres_data.N_unit_to_W_unit(iso_mat['gamut_bg_primary'][0])
corner_points_rgb = iso_mat['cornerPointsRGB'][0]
verts = [list(zip(*gamut_rgb))]

#here we can use either gamut_rgb or corner_points_rgb
centroid = np.mean(corner_points_rgb, axis=1)
# Subtract the centroid to center the points
centered_points = corner_points_rgb - centroid[:,None]
# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(np.transpose(centered_points,(1,0)))

nTheta = 200
sliced_ell_byPlane = np.full((num_grid_pts, num_grid_pts, ndims, nTheta), np.nan)
flat_ell_isoluminant = np.full((num_grid_pts, num_grid_pts,ndims-1, nTheta), np.nan)
for n in range(num_grid_pts):
    for m in range(num_grid_pts):
        ell_params_n = model_pred.params_ell[n][m][0]
        radii_n = ell_params_n['radii']
        center_n = np.reshape(ell_params_n['center'],(-1))
        evecs_n = ell_params_n['evecs']
        sliced_ell_byPlane[n,m], _ = slice_ellipsoid_byPlane(center_n, 
                                                           radii_n, 
                                                           evecs_n, 
                                                           Vt[0], 
                                                           Vt[1],
                                                           num_grid_pts= nTheta)
        flat_ell = color_thres_data.M_RGBTo2DW @ color_thres_data.W_unit_to_N_unit(sliced_ell_byPlane[n,m])
        flat_ell_isoluminant[n,m] = flat_ell[:(ndims-1)]

#create a figure
fig3 = plt.figure(figsize=Plot3D_settings.fig_size, dpi=Plot3D_settings.dpi, constrained_layout=True)
ax3 = fig3.add_subplot(111, projection='3d')

# draw plane (no label)
plane = Poly3DCollection(verts, edgecolor= [0.2,0.2,0.2])
plane.set_facecolor(np.array([[0.5, 0.5, 0.5, 0.3]]))  # RGBA as (1,4) array
ax3.add_collection3d(plane)

# draw ellipses and keep the first line handle
line_handle = None
for n in range(num_grid_pts):
    for m in range(num_grid_pts):
        h, = ax3.plot(*sliced_ell_byPlane[n, m], color='k')
        if line_handle is None:
            line_handle = h

# legend proxies
handles = [
    Patch(facecolor=(0.5, 0.5, 0.5, 0.3), edgecolor='none', label='Isoluminant plane'),
    Line2D([0], [0], color='k', label='Threshold ellipses sliced by the plane'),
]
ax3.legend(handles=handles, loc='lower center', 
           bbox_to_anchor=(0.5, -0.2), fontsize= Plot3D_settings.fontsize - 1)

vis.plot_3D(np.reshape(grid_trans,(num_grid_pts**(ndims-1), ndims)),
            np.reshape(model_pred.fitEll_scaled,(num_grid_pts**(ndims-1), ndims,-1)),
            ax = ax3,
            settings = Plot3D_settings)

# Save the figure as a PDF
#fig3.savefig(os.path.join(output_figDir_fits, f"Sliced_isoluminant_plane_{file_name[:-4]}.pdf"))    
plt.show()

#%% Step 2: Select the bootstrapped data file (choose one as a reference)
# Option 1: 
# 'ELPS_analysis/Experiment_DataFiles/pilot3/sub1/fits/AEPsych_btst'
# 'Fitted_ColorDiscrimination_6dExpt_RGBcube_sub1_decayRate0.4_nBasisDeg5_btst_AEPsych[0].pkl'
input_fileDir_fits_others, file_name_others = select_file_and_get_path()
nTheta = 1000
nDatasets = 10

# Step 2: Initialize storage arrays
# - `params_ell`: Stores ellipse parameters for all grid points across bootstrapped datasets
#   - Ellipse parameters: [x-center, y-center, major axis length, minor axis length, rotation angle]
grid_shape = [num_grid_pts] * (ndims - 1)
params_ell_shape = grid_shape + [nDatasets, 5]
params_ell_btst = np.full(params_ell_shape, np.nan)
sliced_ell_byPlane_btst = np.full((num_grid_pts, num_grid_pts, nDatasets, ndims, nTheta), np.nan)
flat_ell_isoluminant_btst = np.full((num_grid_pts, num_grid_pts, nDatasets, ndims-1, nTheta), np.nan)

# Step 3: Loop through each bootstrap dataset and load data
for r in trange(nDatasets):
    # Replace bootstrap index
    input_fileDir_fits_others_r = input_fileDir_fits_others
    file_name_r = file_name_others.replace('AEPsych[0]', f'AEPsych[{r}]')
    
    # Generate the file name for the current bootstrap dataset
    full_path_others_r = f"{input_fileDir_fits_others_r}/{file_name_r}"
    
    # Load the variables from the current bootstrap dataset
    with open(full_path_others_r, 'rb') as f:
        vars_dict_others = pickled.load(f)
    
    # Use precomputed results if available
    model_pred_r = vars_dict_others['model_pred_Wishart_grid_isoluminant']
    param_ell_r = model_pred_r.params_ell
                
    for i in range(num_grid_pts):
        for j in range(num_grid_pts):
            ell_params_ij = param_ell_r[i][j][0]
            radii_ij = ell_params_ij['radii']
            center_ij = np.reshape(ell_params_ij['center'],(-1))
            evecs_ij = ell_params_ij['evecs']
            sliced_ell_byPlane_btst[i,j,r], _ = slice_ellipsoid_byPlane(center_ij, 
                                                                        radii_ij, 
                                                                        evecs_ij, 
                                                                        Vt[0], 
                                                                        Vt[1],
                                                                        num_grid_pts= nTheta)
            
            flat_ell = color_thres_data.M_RGBTo2DW @ color_thres_data.W_unit_to_N_unit(sliced_ell_byPlane_btst[i,j,r])
            flat_ell_isoluminant_btst[i,j,r] = flat_ell[:(ndims-1)]
            
            #fit an ellipse
            ellipse_ij = EllipseModel()
            #Points need to be in (N,2) array where N is the number of points 
            #Each row is a point [x,y]
            ellipse_ij.estimate(flat_ell_isoluminant_btst[i,j,r].T)
                    
            #Parameters of the fitted ellipse
            xCenter, yCenter, majorAxis, minorAxis, theta_rad = ellipse_ij.params
            theta = np.rad2deg(theta_rad)
            
            params_ell_btst[i,j,r] = [xCenter, yCenter, majorAxis, minorAxis, theta]
            
#Computes the confidence intervals for the model-predicted ellipses at each grid point.
fitEll_min, fitEll_max = find_inner_outer_contours_for_gridRefs(params_ell_btst)
        
#%%           
# -------------------------------------------------------------------------
# Optional: load 4D data
# -------------------------------------------------------------------------
# Prompt user to select a ground truth file
#'ELPS_analysis/Experiment_DataFiles/pilot2/sub1/fits'
#'Fitted_ColorDiscrimination_4dExpt_Isoluminant plane_sub1_decayRate0.5_nBasisDeg5.pkl'
cp_fileDir_fits, cp_file_name = select_file_and_get_path()

# Build the full path to the selected file
cp_full_path = os.path.join(cp_fileDir_fits, cp_file_name)

# Load ground truth variables from the selected file
with open(cp_full_path, 'rb') as f:
    vars_dict_cp = pickled.load(f)

# Otherwise, extract model predictions with matching grid resolution
model_pred_cp = vars_dict_cp['model_pred_Wishart_grid7']
grid_2D = vars_dict_cp['grid7']
num_grid_pts_cp = model_pred_cp.fitEll_scaled.shape[0]
color_thres_data_2D = vars_dict_cp['color_thres_data']
if num_grid_pts_cp != num_grid_pts:
    raise ValueError('The number of grid points does not match across the two loaded datasets.')

cp_ell_vis = model_pred_cp.fitEll_unscaled

#%% Create the output directory if it doesn't exist
pltSettings_base = PlotSettingsBase(fig_dir=output_figDir_fits, fontsize = 8)
# Initialize 2D prediction settings by copying from base and overriding method-specific parameters
pred2D_settings = replace(Plot2DPredSettings(), **pltSettings_base.__dict__)
pred2D_settings = replace(pred2D_settings, 
                          fontsize = 8,
                          visualize_samples= False,
                          visualize_gt = False,
                          visualize_model_estimatedCov = False,
                          flag_rescale_axes_label = False,
                          visualize_model_pred = True,
                          modelpred_alpha = 0.5,
                          modelpred_lw = 0.75,
                          modelpred_lc = 'k',
                          modelpred_ls = '--',
                          modelpred_label = 'Model predictions (4D Expt.)',
                          ticks = np.linspace(-0.7, 0.7,5),
                          title ='Sliced isoluminant plane',
                          fig_name = f"Comparison_w4DExpt_{file_name[:-4]}_decayRate0.4.pdf") 
# Initialize Visualization Class for Wishart Predictions
wishart_pred_vis_wCI = WishartPredictionsVisualization(None,
                                                       model_pred_cp.model, 
                                                       model_pred_cp, 
                                                       color_thres_data_2D,
                                                       settings = pred2D_settings,
                                                       save_fig = True)
# Create figure and axes for plotting
fig, ax = plt.subplots(1, 1, figsize=pred2D_settings.fig_size, dpi=pred2D_settings.dpi)

# plot the confidence interval
for i in range(num_grid_pts):
    for j in range(num_grid_pts):
        cm = color_thres_data.W_unit_to_N_unit(grid_trans[i, j,0])
        if i == 0 and j == 0:
            lbl_btst = f'Bootstrap CI ({nDatasets} datasets; 6D Expt.)' 
            lbl = 'Model predictions (6D Expt.)'
        else:
            lbl_btst = None
            lbl = None
        wishart_pred_vis_wCI.add_CI_ellipses(fitEll_min[i, j], fitEll_max[i, j],
                                             ax=ax, cm=cm, label=lbl_btst, lw_outer = 0,
                                             alpha = 0.7)
        ax.plot(*flat_ell_isoluminant[i,j], color = cm, lw = 0.7, label = lbl)

# Overlay model predictions (joint fits) onto the same axes
wishart_pred_vis_wCI.plot_2D(grid_2D,  ax=ax, settings=pred2D_settings)
