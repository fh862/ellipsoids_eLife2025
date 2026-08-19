#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:36:54 2024

@author: fangfang
"""

import jax
jax.config.update("jax_enable_x64", True)
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import imageio.v2 as imageio
from numpy.polynomial.chebyshev import chebval2d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import os
from analysis.color_thres import color_thresholds
from core import viz

#%%
@dataclass
class PlotSettingsBase:
    fontsize: int = 12
    fontstyle: str = 'Arial' #'Liberation Sans'
    dpi: int = 1024
    fig_dir: str = ''
    
@dataclass
class PlotBasis1DSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (2, 8)
    linewidth: float = 2
    fig_name: str = 'Chebyshev_basis_functions_1D'
    
@dataclass
class PlotCovMatSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (8, 4)
    slc_idx_dim1: int = 0
    slc_idx_dim2: int = 0
    lw_grid: float = 0.5
    lc_grid: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    bds_W_unit: List[float] = field(default_factory=lambda: [-1, 1])
    skip_nticks: int = 2
    plane_2D: str = ''
    cmap: str = 'PRGn'
    flag_remake_cmap: bool = False
    ticks_W: np.ndarray = field(default_factory=lambda: np.array([-0.6, 0, 0.6]))
    cmap_ell: Union[str, np.ndarray] = 'k'
    heatmap_title_list: List[List[str]] = field(default_factory=lambda: [
        [r'$\sigma^2_{dim1}$', r'$\sigma_{(dim1,dim2)}$'],
        [r'$\sigma_{(dim2,dim1)}$', r'$\sigma^2_{dim2}$']
    ])
    scaler_ell: float = 1
    covMat_title: str = '2D plane'
    cmap_bds: List[float] = field(default_factory=list)
    cbar_labelsize: float = 8
    flag_rescale_axes_label: bool = False
    flag_add_horz_vert_lines: bool = True
    fig_name: str = 'CovarianceMatrix_2d'
    
@dataclass
class PlotBasis2DSettings(PlotSettingsBase):
    cmap_bds: List[float] = field(default_factory=list)
    fig_size: Tuple[float, float] = (8, 8)
    fig_name: str = 'Chebyshev_basis_functions_2D'
    cmap: str = 'PRGn'
    flag_remake_cmap: bool = False
    
@dataclass
class PlotBasis3DSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (8, 9)
    xyzlim: List[float] = field(default_factory=lambda: [-1.05, 1.05])
    fig_name: str = 'Chebyshev_basis_function'
    cmap: str = 'PRGn'
    cmap_bds: List[float] = field(default_factory=list)
    flag_remake_cmap: bool = False
    view_anlge: Tuple[float, float] = (20, -75)

@dataclass
class PlotBasis3DBoundaryCubeSettings(PlotBasis3DSettings):
    """Appearance settings for six-face 3D basis-function cubes."""

    fontsize: int = 24
    fig_size: Tuple[float, float] = (12, 12)
    fig_name: str = 'Chebyshev_3D_basis_boundary_cubes'
    figure_title: str = '3D Chebyshev basis functions'
    row_axis_label: Optional[str] = None
    column_axis_label: Optional[str] = None
    panel_axis_label: str = 'Basis function along dimension 3'
    row_titles: Optional[List[str]] = None
    column_titles: Optional[List[str]] = None
    panel_titles: List[str] = field(default_factory=list)
    cmap_bds: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    cube_margin: float = 0.03
    cube_zoom: float = 1.18
    cube_edge_color: str = '0.25'
    cube_edge_linewidth: float = 0.7
    surface_alpha: float = 1.0
    colorbar_shrink: float = 0.72
    colorbar_pad: float = 0.02
    
@dataclass
class PlotWSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (5, 5)
    cmap: str = 'RdBu_r'
    cmap_bds: List[float] = field(default_factory=list)
    add_title: bool = False
    show_colorbar: bool = False
    fig_name: str = 'EstimatedWeightMatrix'
    fig_name_ext: str = ''

@dataclass
class PlotWAllSettings(PlotSettingsBase):
    fig_size: Tuple[float, float] = (6, 3.5)
    jitter: float = 0.1
    marker_alpha: float = 0.8
    marker_size: float = 100
    jitter_seed: int = 0
    marker_color: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [0.3, 0.3, 0.3])
    marker_edgecolor: Union[str, np.ndarray, List[float]] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    yticks: List[float] = field(default_factory=lambda: list(np.linspace(-0.05, 0.05, 3)))
    ybds: List[float] = field(default_factory=lambda: [-0.05, 0.05])
    xlabel: str = 'Max order of the 2D chebyshev polynomial'
    ylabel: str = 'Weight'
    fig_name: str = 'SampledWeightMatrix'
    fig_name_ext: str = ''
    
@dataclass
class PlotUSettings(PlotSettingsBase):
    plane_2D: str = ''
    fig_size: Tuple[float, float] = (5, 4)
    cmap: str = 'PRGn'
    flag_remake_cmap: bool = False
    cmap_bds: List[float] = field(default_factory=list)
    ticks: np.ndarray = field(default_factory=lambda: np.linspace(-0.6, 0.6, 5))
    xyzlim: List[float] = field(default_factory=lambda: [-1.05, 1.05])
    view_anlge: Tuple[float, float] = (20, -75)
    cbar_labelsize: float = 8
    fig_name: str = 'U'
    fig_name_ext: str = ''
    

def add_adaptive_legend(fig, ax, fontsize):
    """Place a deduplicated legend below an unchanged plotting area."""
    handles, labels = ax.get_legend_handles_labels()
    legend_entries = dict(zip(labels, handles))
    labels = list(legend_entries)
    if not labels:
        return None
    handles = [legend_entries[label] for label in labels]

    base_width, base_height = fig.get_size_inches()
    legend_height = 0.2 + 0.035 * fontsize * len(labels)
    figure_height = base_height + legend_height
    fig.set_size_inches(base_width, figure_height)
    fig.tight_layout(rect=(0, legend_height / figure_height, 1, 1))
    fig.canvas.draw()
    axes_bottom = ax.get_tightbbox(fig.canvas.get_renderer()).y0
    legend_top = (axes_bottom - 0.06 * fig.dpi) / fig.bbox.height
    axes_position = ax.get_position()
    legend_center = axes_position.x0 + axes_position.width / 2

    return fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(legend_center, legend_top),
        borderaxespad=0,
        fontsize=fontsize,
    )


#%%
class PlottingTools:
    def __init__(self, settings: PlotSettingsBase, save_fig = False, save_format = 'pdf'):
        self.st = settings
        self.save_fig = save_fig
        self.save_format = save_format
        plt.rcParams['font.family'] = self.st.fontstyle
        plt.rcParams['font.sans-serif'] = self.st.fontstyle
        plt.rcParams['font.size'] = self.st.fontsize
                
    def _save_figure(self, fig, fig_name, bbox_inches=None, pad_inches=0.1):
        """
        Saves the given figure to the specified directory with a provided filename.
    
        Parameters:
            fig (Figure): The matplotlib figure object to save.
            fig_name (str): The filename under which to save the figure.
        """
        # Append '.png' if the filename does not already end with '.png' or '.pdf'
        if (not fig_name.endswith('.png')) and (not fig_name.endswith('.pdf')):
            fmt = '.' + self.save_format
            fig_name += fmt
    
        # Check if the directory exists; if not, create it
        if not os.path.exists(self.st.fig_dir):
            os.makedirs(self.st.fig_dir)

        full_path = os.path.join(self.st.fig_dir, fig_name)
        fig.savefig(full_path, dpi=self.st.dpi,
                    bbox_inches=bbox_inches, pad_inches=pad_inches)

    def _update_axes_limits(self, ax, lim = [-1,1], ndims = 2):
        """
        Sets uniform limits for axes of a plot, extending to 3D if applicable.
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
            lim (list): The limits to set for the x and y (and z if 3D) axes.
        """
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        if ndims == 3: ax.set_zlim(lim)
        
    def _update_axes_labels(self, ax, unit_true_x, unit_show_x, unit_true_y = None,
                            unit_show_y = None, unit_true_z = None, unit_show_z = None,
                            nsteps = 1, ndims = 2):
        """
        Sets ticks and labels for plot axes, adapting to the number of dimensions.
        If there is no input unit_true_y and unit_show_y, it means they are the 
        same as x axis. 
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
            unit_true_x (list): Actual data points for tick positions.
            unit_show_x (list): Values to display at tick positions.
            nsteps (int): Interval for selecting ticks and labels.
            
        """
        #default is 2d plot
        #if there is no input y ticks and y label, we assume they are the same
        #as the x ticks and x label
        if unit_true_y is None and unit_show_y is None:
            unit_true_y = unit_true_x
            unit_show_y = unit_show_x
            
        #if we do not want to have any tickmarks 
        if ~np.any(np.array(unit_true_x)):
            ax.set_xticks([])
            ax.set_yticks([])
        else: #otherwise
            ax.set_xticks(unit_true_x[::nsteps])
            ax.set_yticks(unit_true_y[::nsteps])
            ax.set_xticklabels([f"{x:.2f}" for x in unit_show_x[::nsteps]],\
                               fontsize = self.st.fontsize)
            ax.set_yticklabels([f"{x:.2f}" for x in unit_show_y[::nsteps]],\
                               fontsize = self.st.fontsize)
            
        #if the plot is a 3d
        if ndims == 3:
            if unit_true_z is None and unit_show_z is None:
                unit_true_z = unit_true_x
                unit_show_z = unit_show_x
            if ~np.any(np.array(unit_true_z)):
                ax.set_zticks([])
            else:
                ax.set_zticks(unit_true_z[::nsteps])
                ax.set_zticklabels([f"{x:.2f}" for x in unit_show_z[::nsteps]],\
                                   fontsize = self.st.fontsize)     
        
    def _configure_labels_and_title(self, ax, title = '', ndims =2):
        """
        Configures labels and title for a plot based on predefined plane settings.
        
        Parameters:
            ax (Axes): The matplotlib axes object to modify.
        """
        fs = self.st.fontsize
        if ndims == 2:
            if title in ['GB plane', 'RB plane', 'RG plane']:
                ax.set_xlabel(title[0], fontsize=fs)
                ax.set_ylabel(title[1], fontsize=fs)
                ax.set_title(title, fontsize=fs)
            else:
                # Isoluminant plane
                ax.set_xlabel('Model space dimension 1', fontsize=fs)
                ax.set_ylabel('Model space dimension 2', fontsize=fs)
                if title == '': title = '2D plane'
                ax.set_title(title, fontsize=fs)
        elif ndims == 3:
            if title == 'RGB cube':
                #default
                ax.set_xlabel('R', fontsize=fs)
                ax.set_ylabel('G', fontsize=fs)
                ax.set_zlabel('B', fontsize=fs) 
                ax.set_title(title, fontsize=fs)
            else:
                ax.set_xlabel('Model space dimension 1', fontsize=fs)
                ax.set_ylabel('Model space dimension 2', fontsize=fs)
                ax.set_zlabel('Model space dimension 3', fontsize=fs)
                ax.set_title('3D cube', fontsize=fs)
  
    @staticmethod
    def configure_colormap(data):
        """Choose symmetric color bounds around zero from the data range."""
        max_abs_val = float(np.nanmax(np.abs(data)))
        if not np.isfinite(max_abs_val) or max_abs_val == 0:
            max_abs_val = 1.0
        return [-max_abs_val, max_abs_val]
    
    @staticmethod
    def remake_cmap(cmap, N=256, gamma=0.6):
        """
        Resample a diverging colormap so the light/neutral center band is narrower
        and the saturated ends get more resolution.
    
        This works by sampling the original colormap at non-uniform positions `t(u)`
        that compress values near 0.5 (the center) and expand values near 0 and 1.
    
        Parameters
        ----------
        cmap : str or matplotlib.colors.Colormap
            Base colormap (e.g., "PRGn") or a Colormap object.
        N : int, default 256
            Number of samples used to construct the new colormap.
        gamma : float, default 0.6
            Controls the strength of the center compression.
            - gamma = 1.0: no change (uniform sampling)
            - gamma < 1.0: shrink center / expand extremes (more saturated range)
            - gamma > 1.0: expand center / shrink extremes
    
        Returns
        -------
        matplotlib.colors.ListedColormap
            A new colormap with redistributed color resolution.
        """
        base = mpl.cm.get_cmap(cmap)
        u = np.linspace(0.0, 1.0, N)
    
        # Symmetric nonlinear remapping around 0.5.
        # Using |2u-1|**gamma makes the mapping steeper near the ends for gamma<1,
        # allocating more samples to saturated colors and fewer near the neutral center.
        t = 0.5 + 0.5 * np.sign(u - 0.5) * (np.abs(2 * u - 1) ** gamma)
    
        name = f"{getattr(base, 'name', 'cmap')}_shrinkCenter_g{gamma:g}"
        return mpl.colors.ListedColormap(base(t), name=name)
    
    @staticmethod
    def save_gif(fig_dir, gif_name, fig_name_start, fig_name_end = '.png', fps = 2,
                 reverse_list = False):
        """
        Compiles a sequence of images into a GIF and saves it to the specified directory.
        
        Parameters:
            gif_name (str): The filename for the GIF.
            fig_name_start (str): The beginning pattern of filenames to include in the GIF.
            fig_name_end (str): The ending pattern of filenames to include in the GIF.
            fps (int): Frames per second, defining the speed of the GIF.
        """

        images = [img for img in os.listdir(fig_dir) \
                  if img.startswith(fig_name_start) and img.endswith(fig_name_end)]
        images.sort()  # Sort the images by name (optional)
        image_list = [imageio.imread(f"{fig_dir}/{img}") for img in images]
        if reverse_list: image_list = image_list[::-1]
        # Create a GIF
        # Append '.png' if the filename does not already end with '.png'
        if not gif_name.endswith('.gif'):
            gif_name += '.gif'
        output_path = f"{fig_dir}/{gif_name}"
        imageio.mimsave(output_path, image_list, fps= fps)  
  
#%%
class WishartModelBasicsVisualization(PlottingTools):
    def plot_basis_function_1d(self, degree, grid, cheby_func, 
                               settings: PlotBasis1DSettings, ax = None):
        """
        Plot a series of 1D Chebyshev polynomial basis functions.
    
        This function generates and displays a series of plots, each showing a 
        single Chebyshev polynomial of a specified degree evaluated at provided grid points.
        It allows for customization of the plots through keyword arguments.
    
        Parameters
        ----------
        degree : int
            The degree of the Chebyshev polynomial. This also determines the number of subplots,
            as each degree from 0 to `degree-1` will be plotted.
        grid : array-like, shape (N,)
            The grid points at which the Chebyshev polynomials are evaluated. These points should
            cover the domain of interest, typically [-1, 1] for Chebyshev polynomials.
        cheby_func : array-like, shape (N, degree)
            The values of the Chebyshev polynomials at each point in `grid`. Each column corresponds
            to a polynomial of increasing degree.
        kwargs : dict, optional
            Additional keyword arguments to customize the plots, such as 'linewidth' or 'fig_size'.
            These settings will override the method-specific defaults.

        """
        # Create a figure and a set of subplots with shared x and y axes.
        if ax is None:
            fig, ax = plt.subplots(degree, 1, figsize = settings.fig_size,\
                                             sharex=True, sharey=True)
        else:
            fig = ax.figure
        
        for i in range(degree):
            ax[i].plot(grid,cheby_func[:,i], color = 'k', linewidth = settings.linewidth)
            ax[i].set_aspect('equal')
        plt.show()
        
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)

        return fig, ax

    def plot_2D_covMat(self, grid, cov_fine, cov_grid, settings: PlotCovMatSettings, 
                       ax = None):
        """
        Visualize a 2D field of covariance matrices as both heatmaps and ellipses.

        The figure is split into two parts:

        - Left panel (2×2 subplots):  
          Heatmaps for each element of the 2×2 covariance matrix at each grid
          location:
              (0, 0): σ_x²
              (0, 1): σ_xy
              (1, 0): σ_yx
              (1, 1): σ_y²
        - Right panel (single large subplot):  
          Ellipses corresponding to covariance matrices on a coarser grid, giving
          a geometric view of the local shape and orientation of the noise.

        Parameters
        ----------
        grid : np.array, shape (n_rows, n_cols, 2)
            Grid of reference locations in W units (2D plane).
        cov_fine : np.array, shape (N_rows, N_cols, 2, 2)
            Covariance matrices on a finely sampled grid. These are used for
            the heatmaps on the left.
        cov_grid : np.array, shape (n_rows, n_cols, 2, 2)
            Covariance matrices on a coarser grid. These are used to plot the
            ellipses on the right.
        settings : PlotCovMatSettings
            Configuration for colormaps, ticks, labels, titles, and file saving.
        ax : np.ndarray of Axes, optional
            Optional array of existing axes with shape (2, 4). If None, a new
            figure and axes array is created.

        """
        ndims = 2

        # Configure the colormap bounds from the fine-scale covariances,
        # unless explicit bounds were provided in `settings`.
        cmap_bds = settings.cmap_bds if settings.cmap_bds else self.configure_colormap(cov_fine)
        
        if settings.flag_remake_cmap: cmap = PlottingTools.remake_cmap(settings.cmap);
        else: cmap = settings.cmap

        # Number of finely sampled and coarsely sampled grid points.
        num_grid_fine_rows, num_grid_fine_cols = cov_fine.shape[:2]
        num_grid_rows, num_grid_cols = cov_grid.shape[:2]

        if grid.shape[:2] != (num_grid_rows, num_grid_cols):
            raise ValueError(
                "`grid` and `cov_grid` must share the same first two dimensions. "
                f"Got grid.shape[:2]={grid.shape[:2]} and "
                f"cov_grid.shape[:2]={cov_grid.shape[:2]}."
            )

        # Convert tick locations from W units (model space) to N units (plot space).
        ticks_N = color_thresholds.W_unit_to_N_unit(settings.ticks_W)
        
        if ax is None:
            # Create a 2×4 grid of subplots.
            # The first 2×2 block is used for heatmaps; the remaining 4 axes
            # will be deleted and replaced by a single large ellipse panel.
            fig, ax = plt.subplots(2, 4, figsize=settings.fig_size, 
                                   dpi = settings.dpi,
                                   sharex=True, sharey=True)
        else:
            # Expect an array-like of Axes with shape (2,4)
            axs = ax
            if not isinstance(axs, np.ndarray):
                raise TypeError("`ax` must be a numpy.ndarray of Axes with shape (2,4), or None.")
            if axs.shape != (2, 4):
                raise ValueError(f"`ax` must have shape (2,4). Got {axs.shape}.")
            fig = axs.ravel()[0].figure
                
        for i in range(ndims):
            for j in range(ndims):      
                # Heatmap for the (i, j) entry of the covariance matrix across the fine grid.
                im = ax[i, j].imshow(cov_fine[:,:,i,j], 
                                     origin='lower',
                                     cmap = cmap,
                                     vmin = cmap_bds[0], 
                                     vmax = cmap_bds[1]
                                     )

                # Optionally draw crosshair lines at a selected grid location.
                if settings.flag_add_horz_vert_lines:
                    
                    # Extract the selected coarse-grid point in W units.
                    grid_slc = grid[settings.slc_idx_dim1, settings.slc_idx_dim2]
                    
                    # Map from W units to N units and then to pixel indices of the fine grid.
                    grid_norm = color_thresholds.W_unit_to_N_unit(grid_slc) * np.array(
                        [num_grid_fine_cols, num_grid_fine_rows]
                    )
                    xv_grid, yv_grid = grid_norm

                    # Horizontal line through the selected point.
                    ax[i, j].plot([0, num_grid_fine_cols],
                                  [yv_grid, yv_grid],
                                  c = settings.lc_grid,
                                  lw = settings.lw_grid
                                  )
                    # Vertical line through the selected point.
                    ax[i, j].plot([xv_grid, xv_grid],
                                  [0, num_grid_fine_rows], 
                                  c = settings.lc_grid,
                                  lw = settings.lw_grid
                                  )
                    # Mark the intersection of the crosshair.
                    ax[i, j].scatter(xv_grid, yv_grid, c = 'k', s = 10)
                    
                # Set axis ticks and labels in either W units or N units.
                if settings.flag_rescale_axes_label:
                    self._update_axes_labels(ax[i,j], 
                                             ticks_N * num_grid_fine_cols,
                                             ticks_N,
                                             ticks_N * num_grid_fine_rows,
                                             ticks_N, 
                                             nsteps= 1
                                             )
                else:
                    self._update_axes_labels(ax[i,j], 
                                             ticks_N * num_grid_fine_cols,
                                             settings.ticks_W, 
                                             ticks_N * num_grid_fine_rows,
                                             settings.ticks_W, 
                                             nsteps = 1
                                             )  
                # Restrict to the full fine-grid extent.
                ax[i, j].set_xlim(0, num_grid_fine_cols - 1)
                ax[i, j].set_ylim(0, num_grid_fine_rows - 1)

                # Title for each covariance component (e.g., σ_x², σ_xy, σ_y²).
                ax[i, j].set_title(settings.heatmap_title_list[i][j])
        
        # Remove the four axes in the right half; they will be replaced with
        # a single combined panel for the ellipses.
        plt.delaxes(ax[0, 2])
        plt.delaxes(ax[0, 3])
        plt.delaxes(ax[1, 2])
        plt.delaxes(ax[1, 3])
        
        # Shared colorbar for the heatmaps.
        cbar_ax = fig.add_axes([0.065, 0.1, 0.4, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=settings.cbar_labelsize)
        
        # Add a large subplot that spans the right half of the figure
        # (replacing the four deleted axes).
        ax_ell = fig.add_subplot(1, 2, 2)  
        # Loop over the coarse grid and plot ellipses for each covariance.
        for row in range(num_grid_rows):
            for col in range(num_grid_cols):
                # Choose ellipse color: either a single color or a per-cell colormap.
                if isinstance(settings.cmap_ell, str):
                    cmap_ell = settings.cmap_ell
                else:
                    cmap_ell = settings.cmap_ell[row, col]
                
                # Only plot ellipses up to and including the selected slice index.
                if row < settings.slc_idx_dim1 or (row == settings.slc_idx_dim1 \
                                                 and col <= settings.slc_idx_dim2):
                    viz.plot_ellipse(ax_ell,
                                     grid[row,col],
                                     cov_grid[row,col]* settings.scaler_ell,
                                     c= cmap_ell
                                     )
        # Configure axes for the ellipse panel: limits, tick labels, and titles.
        self._update_axes_limits(ax_ell, settings.bds_W_unit) 
        if settings.flag_rescale_axes_label:
            self._update_axes_labels(ax_ell, settings.ticks_W, ticks_N, nsteps = 1)
        else:
            self._update_axes_labels(ax_ell, settings.ticks_W, settings.ticks_W, nsteps = 1)  
        self._configure_labels_and_title(ax_ell)
        ax_ell.grid(True, alpha=0.5)
        ax_ell.set_aspect('equal')
        ax_ell.set_title(settings.covMat_title)
        # Show the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.2)
        #plt.show()
        fig_counter = settings.slc_idx_dim1 * num_grid_cols + settings.slc_idx_dim2
        fig_name = f"{settings.fig_name}_{fig_counter:02d}"
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, fig_name)
            
        return fig, ax, ax_ell
    
            
#%%     
    def plot_basis_function_2D(self, degree, grid, settings: PlotBasis2DSettings, 
                               ax = None):
        """
        Plot 2D Chebyshev basis functions on a specified grid.
    
        This function generates a grid of subplots where each subplot represents 
        one of the basis functions for a 2D Chebyshev polynomial of specified 
        degrees. It demonstrates the effect of individual polynomial terms in 
        two dimensions, each identified by a pair of indices (i, j).

        Parameters
        ----------
        degree : int
            The degree of the Chebyshev polynomial. This also determines the number of subplots,
            as each degree from 0 to `degree-1` will be plotted.
        grid : array-like, shape (N,)
            The grid points at which the Chebyshev polynomials are evaluated. These points should
            cover the domain of interest, typically [-1, 1] for Chebyshev polynomials.

        """        
        # Create a 2D mesh grid using the provided 1D grid array.
        xg, yg = np.meshgrid(grid, grid)
        # Initialize a grid for storing coefficients of the 2D polynomials.
        cg = np.zeros((degree, degree))
        basis_values = np.zeros((grid.shape[0], grid.shape[0], degree, degree))
        for i in range(degree):
            for j in range(degree):
                cg[i, j] = 1.0
                basis_values[:, :, i, j] = chebval2d(xg, yg, cg)
                cg[i, j] = 0.0
        cmap_bds = settings.cmap_bds if settings.cmap_bds else self.configure_colormap(basis_values)
        
        if ax is None:
            # Create a figure with subplots arranged in a square grid.
            fig, ax = plt.subplots(degree, degree, figsize= settings.fig_size,
                                     sharex=True, sharey=True)
        else:
            fig = ax.figure

        for i in range(degree):
            for j in range(degree):
                # Display the result as an image in the corresponding subplot.
                ax[i, j].imshow(basis_values[:, :, i, j], origin='lower',
                                  cmap = settings.cmap, 
                                  vmin = cmap_bds[0],
                                  vmax = cmap_bds[1])
                
                # Update axis labels and limits to make the plots cleaner.
                self._update_axes_labels(ax[i,j], [], [])
                self._update_axes_limits(ax[i,j], [0,grid.shape[0]-1])
                
                # Set a title for each subplot to indicate the polynomial degrees.
                ax[i, j].set_title(f"({i}, {j})")
        plt.tight_layout()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, settings.fig_name)
        
        return fig, ax
    
#%%
    def plot_basis_functions_3D(self, XG, YG, ZG, M, settings: PlotBasis3DSettings):
        """
            Visualizes selected slices of 3D Chebyshev basis functions over time.
        
            Due to the complexity of visualizing high-dimensional data, this function simplifies
            the representation by displaying one 2D slice at a time from the 3D 
            Chebyshev basis functions. Each slice is treated as a time step, providing 
            a series of 2D plots that represent how the basis functions evolve over 
            the third dimension, conceptualized as time.
            
            This script is also used to visualize selected slices of weigthed sum
            of basis functions (U), as well as 3D covariance matrices (Sigmas)
        
            Parameters
            ----------
            XG, YG, ZG : array-like, shape (N, N, N)
                3D grids representing the x, y, and z coordinates in the cube, respectively. 
                These grids define the points at which the basis functions are evaluated.
            M : array-like, shape (N, N, N, degree, degree)
                The values of the basis functions evaluated at every point in the 3D grid 
                (XG, YG, ZG) for each combination of polynomial degrees up to the specified 
                degree.

        """
        colormap = plt.get_cmap(settings.cmap)
        cmap_bds = settings.cmap_bds if settings.cmap_bds else self.configure_colormap(M)
        cmap_min, cmap_max = cmap_bds
        cmap_span = cmap_max - cmap_min
        
        nbins = M.shape[0]  # Number of bins (slices) in the third dimension.
        ndim1 = M.shape[-2]
        ndim2 = M.shape[-1]
        for l in range(nbins):
            # Create a new figure with 3D subplots for each time point.
            fig, ax = plt.subplots(ndim1, ndim2, dpi = settings.dpi,
                                   figsize = settings.fig_size,
                                   subplot_kw={'projection': '3d'})
            for i in range(ndim1):
                for j in range(ndim2): 
                    normalized = (M[:, :, l, i, j] - cmap_min) / (cmap_span + 1e-10)
                    normalized = np.clip(normalized, 0, 1)
                    # Plot the basis functions.
                    ax[i, j].plot_surface(XG[:,:,l], ZG[:,:,l], YG[:,:,l],\
                        facecolors=colormap(normalized),
                        rstride=1, cstride=1)
                    # Set aspect ratio and limits for each subplot.
                    self._update_axes_labels(ax[i, j], [], [],
                                             unit_true_z = [], 
                                             ndims = 3)
                    self._update_axes_limits(ax[i,j], settings.xyzlim, 
                                             ndims = 3)
                    ax[i, j].view_init(*settings.view_anlge)
                    ax[i, j].set_aspect('equal')
                    ax[i, j].set_autoscale_on(False)
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,\
                                wspace=-0.1, hspace=-0.1)
            plt.show()
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig, f"{settings.fig_name}_slice{l:02}")

    def plot_basis_functions_3D_boundary_cubes(
            self, XG, YG, ZG, M,
            settings: PlotBasis3DBoundaryCubeSettings):
        """Plot tensor-product 3D basis functions on cube boundaries.

        For a three-axis component array, one figure is generated for each
        component along its final axis. For a two-axis component array, one
        figure is generated. Within each figure, the first two component axes
        select subplot rows and columns. Each subplot is a cube whose six
        exterior faces are colored by the corresponding 3D field values. The
        front face therefore shows model-space dimensions 1 and 2, while cube
        depth represents model-space dimension 3.

        Parameters
        ----------
        XG, YG, ZG : array-like, shape (N, N, N)
            Structured coordinate grids for model-space dimensions 1, 2, and
            3. Both ``np.meshgrid`` indexing conventions are supported.
        M : array-like, shape (N, N, N, rows, columns[, panels])
            Values of each plotted field on the grid. A final ``panels`` axis
            is optional.
        settings : PlotBasis3DBoundaryCubeSettings
            Figure, colormap, camera, cube, and output settings. ``cmap_bds``
            sets the shared color limits for every generated figure.

        Yields
        ------
        fig, axes
            A Matplotlib figure and its ``(rows, columns)`` axes array for each
            panel, or a single figure when ``M`` has no panel axis.
        """
        coordinate_grids = tuple(np.asarray(grid) for grid in (XG, YG, ZG))
        basis_values = np.asarray(M)
        grid_shape = coordinate_grids[0].shape

        if any(grid.ndim != 3 or grid.shape != grid_shape
               for grid in coordinate_grids):
            raise ValueError("XG, YG, and ZG must be matching 3D grids.")
        has_panel_axis = basis_values.ndim == 6
        if basis_values.ndim == 5:
            basis_values = basis_values[..., np.newaxis]
        if basis_values.ndim != 6 or basis_values.shape[:3] != grid_shape:
            raise ValueError(
                "M must have shape (*XG.shape, rows, columns[, panels]); "
                f"got XG.shape={grid_shape} and M.shape={basis_values.shape}."
            )

        num_rows, num_columns, num_panels = basis_values.shape[-3:]
        if (settings.row_titles is not None
                and len(settings.row_titles) != num_rows):
            raise ValueError(
                "settings.row_titles must contain one title per subplot row."
            )
        if (settings.column_titles is not None
                and len(settings.column_titles) != num_columns):
            raise ValueError(
                "settings.column_titles must contain one title per subplot column."
            )
        coordinate_axes, spatial_order = self._structured_mesh_axes(
            coordinate_grids
        )
        display_axes = (
            coordinate_axes[0],
            coordinate_axes[2],
            coordinate_axes[1],
        )
        surface_grids = self._cube_surface_grids(display_axes)
        surface_quads = self._cube_surface_quads(surface_grids)
        cube_edges = self._cube_edges(display_axes)

        cmap_bds = np.asarray(settings.cmap_bds, dtype=float)
        if (cmap_bds.shape != (2,)
                or not np.all(np.isfinite(cmap_bds))
                or cmap_bds[0] >= cmap_bds[1]):
            raise ValueError(
                "settings.cmap_bds must contain two finite, increasing bounds."
            )
        cmap = (
            self.remake_cmap(settings.cmap)
            if settings.flag_remake_cmap
            else plt.get_cmap(settings.cmap)
        )
        norm = mpl.colors.Normalize(*cmap_bds)
        colorbar_ticks = np.linspace(*cmap_bds, 5)

        for panel_index in range(num_panels):
            fig, axes = plt.subplots(
                num_rows,
                num_columns,
                figsize=settings.fig_size,
                dpi=settings.dpi,
                subplot_kw={'projection': '3d'},
                constrained_layout=True,
            )
            axes = np.asarray(axes, dtype=object).reshape(
                num_rows, num_columns
            )

            for row_index in range(num_rows):
                for column_index in range(num_columns):
                    ax = axes[row_index, column_index]
                    basis_xyz = np.transpose(
                        basis_values[
                            ..., row_index, column_index, panel_index
                        ],
                        spatial_order,
                    )
                    surface_values = self._basis_cube_surface_values(basis_xyz)
                    facecolors = np.concatenate([
                        cmap(norm(self._cell_means(face))).reshape(-1, 4)
                        for face in surface_values
                    ])
                    facecolors[:, 3] = settings.surface_alpha

                    ax.add_collection3d(Poly3DCollection(
                        surface_quads,
                        facecolors=facecolors,
                        edgecolors='none',
                        linewidths=0,
                        antialiaseds=False,
                    ))
                    ax.add_collection3d(Line3DCollection(
                        cube_edges,
                        colors=settings.cube_edge_color,
                        linewidths=settings.cube_edge_linewidth,
                    ))
                    self._format_basis_cube_axis(ax, display_axes, settings)

                    if row_index == 0 and settings.column_titles is not None:
                        ax.set_title(
                            settings.column_titles[column_index],
                            fontsize=settings.fontsize,
                        )
                    if column_index == 0 and settings.row_titles is not None:
                        ax.text2D(
                            -0.08,
                            0.5,
                            settings.row_titles[row_index],
                            transform=ax.transAxes,
                            rotation=90,
                            ha='center',
                            va='center',
                            fontsize=settings.fontsize,
                        )

            figure_title = settings.figure_title
            if has_panel_axis:
                panel_title = (
                    settings.panel_titles[panel_index]
                    if panel_index < len(settings.panel_titles)
                    else rf'$T_{{{panel_index}}}$'
                )
                figure_title += (
                    f"\n{settings.panel_axis_label}: {panel_title} "
                    f"({panel_index + 1}/{num_panels})"
                )
            fig.suptitle(figure_title, fontsize=settings.fontsize + 3)
            if settings.column_axis_label is not None:
                fig.supxlabel(
                    settings.column_axis_label,
                    fontsize=settings.fontsize + 1,
                )
            if settings.row_axis_label is not None:
                fig.supylabel(
                    settings.row_axis_label,
                    fontsize=settings.fontsize + 1,
                )
            colorbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axes.ravel().tolist(),
                shrink=settings.colorbar_shrink,
                pad=settings.colorbar_pad,
                aspect=30,
                ticks=colorbar_ticks,
            )
            colorbar.ax.tick_params(labelsize=settings.fontsize)

            if settings.fig_dir and self.save_fig:
                panel_suffix = (
                    f"_dim3_{panel_index:02d}" if has_panel_axis else ""
                )
                self._save_figure(
                    fig,
                    f"{settings.fig_name}{panel_suffix}",
                )
            yield fig, axes

    @staticmethod
    def _structured_mesh_axes(coordinate_grids):
        """Return coordinate vectors and array-axis order for a 3D mesh."""
        spatial_order = []
        coordinate_axes = []
        for coordinate_grid in coordinate_grids:
            variation = [
                np.max(np.abs(np.diff(coordinate_grid, axis=axis)))
                if coordinate_grid.shape[axis] > 1 else 0.0
                for axis in range(3)
            ]
            varying_axis = int(np.argmax(variation))
            if variation[varying_axis] == 0:
                raise ValueError("Each coordinate grid must vary along one axis.")
            spatial_order.append(varying_axis)

            index = [0, 0, 0]
            index[varying_axis] = slice(None)
            coordinate_axes.append(coordinate_grid[tuple(index)])

        if len(set(spatial_order)) != 3:
            raise ValueError(
                "XG, YG, and ZG must vary along three different array axes."
            )
        return tuple(coordinate_axes), tuple(spatial_order)

    @staticmethod
    def _cube_surface_grids(display_axes):
        """Construct the six boundary grids in displayed (x, z, y) order."""
        axis_x, axis_y, axis_z = display_axes
        grid_y, grid_z = np.meshgrid(axis_y, axis_z, indexing='ij')
        grid_x_y, grid_z_y = np.meshgrid(axis_x, axis_z, indexing='ij')
        grid_x_z, grid_y_z = np.meshgrid(axis_x, axis_y, indexing='ij')
        return (
            np.stack((np.full_like(grid_y, axis_x[0]), grid_y, grid_z), -1),
            np.stack((np.full_like(grid_y, axis_x[-1]), grid_y, grid_z), -1),
            np.stack((grid_x_y, np.full_like(grid_x_y, axis_y[0]), grid_z_y), -1),
            np.stack((grid_x_y, np.full_like(grid_x_y, axis_y[-1]), grid_z_y), -1),
            np.stack((grid_x_z, grid_y_z, np.full_like(grid_x_z, axis_z[0])), -1),
            np.stack((grid_x_z, grid_y_z, np.full_like(grid_x_z, axis_z[-1])), -1),
        )

    @staticmethod
    def _cube_surface_quads(surface_grids):
        return np.concatenate([
            np.stack((
                surface[:-1, :-1],
                surface[1:, :-1],
                surface[1:, 1:],
                surface[:-1, 1:],
            ), axis=2).reshape(-1, 4, 3)
            for surface in surface_grids
        ])

    @staticmethod
    def _basis_cube_surface_values(basis_xyz):
        """Match model-space values to displayed cube faces (x, z, y)."""
        return (
            basis_xyz[0, :, :].T,
            basis_xyz[-1, :, :].T,
            basis_xyz[:, :, 0],
            basis_xyz[:, :, -1],
            basis_xyz[:, 0, :],
            basis_xyz[:, -1, :],
        )

    @staticmethod
    def _cell_means(values):
        return 0.25 * (
            values[:-1, :-1]
            + values[1:, :-1]
            + values[1:, 1:]
            + values[:-1, 1:]
        )

    @staticmethod
    def _cube_edges(display_axes):
        bounds = [(axis[0], axis[-1]) for axis in display_axes]
        corners = np.array([
            [bounds[0][x], bounds[1][y], bounds[2][z]]
            for x in range(2) for y in range(2) for z in range(2)
        ])
        edge_indices = (
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        )
        return tuple(corners[list(indices)] for indices in edge_indices)

    @staticmethod
    def _format_basis_cube_axis(ax, display_axes, settings):
        for axis, setter in zip(
                display_axes,
                (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            lower, upper = float(np.min(axis)), float(np.max(axis))
            padding = settings.cube_margin * (upper - lower)
            setter((lower - padding, upper + padding))
        ax.set_box_aspect((1, 1, 1), zoom=settings.cube_zoom)
        ax.set_proj_type('ortho')
        ax.view_init(*settings.view_anlge)
        ax.set_axis_off()

#%% 
    def plot_W_selected_slice(self, W, settings: PlotWSettings, basis_orders = None,
                              slc_slice=[0]):
        """
        Plots selected slices of the weight matrix for 2D or 3D Chebyshev polynomial basis functions.
    
        This function visualizes slices of the weight matrix applied to Chebyshev basis functions,
        emphasizing the variation of weights across different polynomial orders. Each plotted figure 
        shows a 2D slice of the weight matrix for a specific degree of polynomial, annotated with 
        the highest polynomial order of the basis functions at each point.
    
        Parameters
        ----------
        degree : int
            The degree of the polynomial.
        W : array
            The weight matrix that modifies the basis functions, with the following shapes:
            - For 2D: (degree, degree, ndims, ndims + extra_dims)
            - For 3D: (degree, degree, degree, ndims, ndims + extra_dims)
        basis_orders : array, optional
            Specifies the highest polynomial order for each basis function, used for annotation in plots, with the following shapes:
            - For 2D: (degree, degree, ndims, ndims + extra_dims)
            - For 3D: (degree, degree, degree, ndims, ndims + extra_dims)
        slc_slice : list of int, optional
            Indices specifying the slice of the last two dimensions of W to be visualized. Defaults to [0].
    
        """
        cmap_bds = settings.cmap_bds if settings.cmap_bds else self.configure_colormap(W)
        
        degree = W.shape[0]
        
        # Reshape the weight matrix and basis orders for consistent indexing
        W_org = np.reshape(W, (degree, degree, -1))

        for i in slc_slice:
            fig, ax = plt.subplots(1, 1, figsize= settings.fig_size,\
                                   sharex=True, sharey=True)
            cax = ax.imshow(W_org[..., i],
                            cmap=settings.cmap,
                            vmin=cmap_bds[0],
                            vmax=cmap_bds[1])
    
            if settings.show_colorbar:
                # Add the colorbar to the right of the main axis
                cbar = fig.colorbar(cax, ax=ax)
                cbar.ax.tick_params(labelsize=settings.fontsize)  # Adjust font size of colorbar ticks
    
            # Set the aspect of the main axis to be square
            ax.set_aspect('equal')
            
            # Annotate each cell in the plot with the highest polynomial order, if provided
            if basis_orders is not None:
                basis_orders_org = np.reshape(basis_orders, (degree, degree, -1))
                for j in range(degree):
                    for k in range(degree):
                            # Display text over the image
                            ax.text(j, k, str(basis_orders_org[j,k,i]),
                                    color='black',\
                                    fontsize=settings.fontsize,
                                    ha='center', 
                                    va='center')
            self._update_axes_labels(ax, [],[])
            if settings.add_title:
                ax.set_title(f'selected slice index: {i}')
            ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            if settings.fig_dir and self.save_fig:
                self._save_figure(fig, f"{settings.fig_name}{settings.fig_name_ext}"+ \
                                  f'_degree{degree}_{i}')

    def plot_W_all(self, W, basis_orders, settings: PlotWAllSettings, ax = None):
        """
        Plots a scatter plot of the weights from a weight matrix as a function of 
            the maximum order of the basis functions.
        
        Parameters
        ----------
        W : array
            The weight matrix that modifies the basis functions, with the following shapes:
            - For 2D: (degree, degree, ndims, ndims + extra_dims)
            - For 3D: (degree, degree, degree, ndims, ndims + extra_dims)
        basis_orders : array, optional
            Specifies the highest polynomial order for each basis function, used for 
            annotation in plots, with the following shapes:
            - For 2D: (degree, degree, ndims, ndims + extra_dims)
            - For 3D: (degree, degree, degree, ndims, ndims + extra_dims)
    
        """
        if ax is None:
            # Create a figure with subplots arranged in a square grid.
            fig, ax = plt.subplots(1, 1, figsize= settings.fig_size,
                                   dpi = settings.dpi)
        else:
            fig = ax.figure
        
        np.random.seed(settings.jitter_seed)
        #flatten the data
        W_org = W.flatten()
        basis_orders_org = basis_orders.flatten()
        # Generate jitter to add randomness to the x-axis positions of the points
        jitter = np.random.randn(len(W_org)) * settings.jitter
        
        # Create a scatter plot of the weights vs. the maximum order of the basis functions
        ax.scatter(basis_orders_org + jitter, W_org,
                    s =settings.marker_size,
                    color = settings.marker_color,
                    edgecolor = settings.marker_edgecolor, 
                    alpha = settings.marker_alpha)
        # Add a horizontal line at y = 0 to the plot
        ax.plot([0,np.max(basis_orders_org)], [0,0],
                color = [0.5,0.5,0.5],
                linestyle = '--', 
                linewidth = 1)
        # Set y-axis ticks and limits
        ax.set_yticks(settings.yticks)
        ax.set_ylim(settings.ybds)
        # Add a grid to the plot for better readability
        ax.grid(True, alpha=0.3)
        # Set x and y-axis labels
        ax.set_xlabel(settings.xlabel)
        ax.set_ylabel(settings.ylabel)
        plt.tight_layout()
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, f"{settings.fig_name}{settings.fig_name_ext}"+
                              f'_degree{W.shape[0]}')
        return fig, ax

    def plot_U_2D(self, U, settings: PlotUSettings, ax = None):
        """
        Visualizes the weighted sum of 2D Chebyshev basis functions.
    
        Parameters
        ----------
        U : array, shape: (N x N x ndims x (ndims + extra_dims))
            The input array representing the weighted sum of 2D Chebyshev basis functions.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axes for the plot. If None, new axes will be created.
        kwargs : dict
            Additional keyword arguments to override default plotting settings.
    
        """
        cmap_bds = settings.cmap_bds if settings.cmap_bds else self.configure_colormap(U)
        
        # Extract the number of finely sampled grid points.
        num_grid_fine = U.shape[0]
        nRows, nCols = U.shape[-2:] # Number of rows and columns in the basis function.
        
        if ax is None:
            # Create a figure with subplots arranged in a 2x3 grid if no axes are provided.
            fig, ax = plt.subplots(nRows, nCols, figsize= settings.fig_size,
                                   dpi = settings.dpi, sharex=True, sharey=True)
        else:
            fig = ax.figure
        
        # Iterate over each element in the basis function and plot it.
        for i in range(nRows):
            for j in range(nCols):      
                # Plot the 2D basis function component using a heatmap.
                im = ax[i, j].imshow(U[:,:,i,j], 
                                     cmap = settings.cmap,
                                     vmin = cmap_bds[0], 
                                     vmax = cmap_bds[1])
                
                # Update axis labels with appropriate tick marks and limits.
                self._update_axes_labels(ax[i,j], 
                                         color_thresholds.W_unit_to_N_unit(\
                                            settings.ticks)*num_grid_fine,
                                         settings.ticks
                                         )

               # Ensure the axis limits are set properly and maintain aspect ratio.
                self._update_axes_limits(ax[i,j], [0,num_grid_fine-1])
                ax[i,j].set_aspect('equal')
        
        # Add a colorbar below the subplots.
        cbar_ax = fig.add_axes([0.3, 0, 0.4, 0.02])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=settings.cbar_labelsize)
        
        # Adjust the layout of the subplots for better spacing and display the figure.
        plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.2)
        plt.tight_layout()
        plt.show()
        if settings.fig_dir and self.save_fig:
            self._save_figure(fig, f"{settings.fig_name}{settings.fig_name_ext}")
            
        return fig, ax
        
        
