#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 22:32:10 2024

@author: fangfang
"""

import numpy as np
import time  
import threading
import jax.numpy as jnp
import io
import configparser
import warnings
from core import oddity_task

#%%
class SimulateTrialGivenWishart:
    """Simulate threshold oddity trials proposed by AEPsych.

    The base class implements the shared AEPsych loop, timeout monitoring,
    value scaling, and trial-data storage. Subclasses can specialize the
    stimulus/response processing through ``_process_trial`` and
    ``_simulate_pregenerated_trial`` and can replace ``_trial_data_fields``
    to define their storage schema.

    All configuration strings are assumed to use the same parameter names,
    strategy names, and per-strategy trial quotas; those fields are read from
    the first configuration string.
    """

    # This schema drives list initialization, trial updates, and final stacking.
    _trial_data_fields = (
        "xref",
        "x1",
        "Uref",
        "U1",
        "signed_diff",
        "pX1",
        "binaryResp",
    )
    
    def __init__(self, expt_dim, config_all, gt_Wishart, ref = None, 
                 pseudo_randomize = False, pseudo_randomize_seed = None, 
                 val_scaler = None, customized_val_scaler = None):
        """
        Initialize a threshold-trial simulator and its storage arrays.
        
        Args:
            expt_dim (int): AEPsych parameterization dimension. Supported
                values are 2 through 6. The 5D case uses four sampled color
                parameters plus one fixed ancillary value selected from
                ``ref``.
            config_all (list of str): One AEPsych configuration string per
                experimental condition.
            gt_Wishart (object): Wishart model used to simulate responses.
            ref (sequence, optional): Fixed references for 2D/3D experiments,
                or fixed ancillary values for 5D experiments. It is required
                when either of those parameterizations is used.
            pseudo_randomize (bool): Whether to shuffle condition order
                independently at every trial position.
            pseudo_randomize_seed (int, optional): Seed for condition-order
                shuffling. It does not seed simulated binary responses.
            val_scaler (sequence, optional): One scaling factor per AEPsych
                strategy. Short or long sequences are padded or truncated.
            customized_val_scaler (sequence, optional): Trial-level scaling
                factors. When supplied, these take priority over ``val_scaler``.
        
        Raises:
            ValueError: If ``expt_dim`` is outside 2--6 or if a required fixed
                reference is missing for a 2D/3D experiment.
        """
        
        if expt_dim not in list(range(2,7)):
            raise ValueError("Color discrimination experiment must be 2, 3, 4, 5, or 6D.")
        self.expt_dim   = expt_dim
        self.gt_Wishart = gt_Wishart
        self.config_all = config_all
        self.numConfig  = len(config_all)
        self.ref        = ref
        
        # Validate the need for reference stimuli in 2D and 3D setups
        if (self.expt_dim == 2 or self.expt_dim == 3) and self.ref is None:
            raise ValueError("If the experiment has fixed ref stimulus, you need to specify ref stimuli!")
        
        # All conditions are expected to share this structure, so parsing the
        # first configuration is sufficient for the common metadata.
        self.parnames       = self._extract_field_vals('common','parnames', strsplit = True)
        self.strat_names    = self._extract_field_vals('common','strategy_names', strsplit = True)
        
        # Read per-strategy trial quotas from the config.
        #
        # Normal workflow uses `min_asks` (strategy advances after generating N asks).
        # DB-switch / preload workflow uses `min_total_tells` (strategy advances after
        # accumulating N tells, including preloaded historical trials).
        #
        # We support both by trying `min_asks` first and falling back to
        # `min_total_tells` if the field is not present in the config.
        try:
            self.nTrials_strat = [
                int(self._extract_field_vals(s, "min_asks")) for s in self.strat_names
            ]
        except Exception:
            self.nTrials_strat = [
                int(self._extract_field_vals(s, "min_total_tells")) for s in self.strat_names
            ]
                
        self.nTrials_cumsum = np.cumsum(np.array(self.nTrials_strat))
        self.nTrials        = self.nTrials_cumsum[-1]
        
        # Each column maps scheduling lanes to condition indices. The subclass
        # overrides the number of columns for a longer interleaved sequence.
        self.pseudo_randomize = pseudo_randomize
        self.pseudo_randomize_seed = pseudo_randomize_seed
        self.pseudo_order = self._create_pseudorandom_order()
        
        # Initialize lists to store trial data
        self._init_trial_lists()
        
        # Customized trial-level scalers take priority over strategy-level scalers.
        if customized_val_scaler is not None:
            self.customized_val_scaler = customized_val_scaler
            self.val_scaler = None
        else:
            self.customized_val_scaler = None
        
            # This method validates val_scaler and assigns the result to
            # self.val_scaler. If val_scaler is None, it creates a list of ones.
            self._validate_val_scaler(val_scaler)
    
    def _extract_field_vals(self, section, field, strsplit = False):
        """
        Extract a field from the first AEPsych configuration string.
        
        Args:
            section (str): Section name in the configuration.
            field (str): Field name from which to extract the values.
            strsplit (bool): If True, splits the string into a list.
        
        Returns:
            str or list[str]: The raw field value, or a comma-separated list
            parsed from bracket notation when ``strsplit`` is True.
        """
    
        # Create a new ConfigParser object
        config_parser = configparser.ConfigParser()
        # Read the configuration from the string
        config_parser.read_file(io.StringIO(self.config_all[0]))
        #retrieve the field
        result_string = config_parser.get(section, field)
        #if we want to split the string to a list of string
        #e.g., '[strat1, strat2]' -> ['strat1', 'strat2']
        if strsplit:
            # Remove square brackets and split by comma
            result_list = result_string.strip('[]').split(',')
            
            # Strip any leading or trailing whitespace from each element
            result_list = [item.strip() for item in result_list]
            return result_list
        else:
            return result_string
    
    def _create_pseudorandom_order(self):
        """
        Generate a 2D array specifying the order of trial configurations for each trial set.
    
        If `self.pseudo_randomize` is False, the configurations are repeated in the same
        fixed order across all trial sets (i.e., sequential). If True, each column
        (trial set) is independently shuffled to introduce randomization.
    
        If `self.pseudo_randomize_seed` is provided, it is used to seed the random number
        generator for reproducible shuffling.
    
        Returns:
            np.ndarray: A 2D array of shape (numConfig, nTrials), where each column contains
                        the order of configurations for a given trial set.
        """
        order_temp = np.array(list(range(self.numConfig)))
        pseudo_order = np.tile(order_temp[:, np.newaxis], (1, self.nTrials))
        if not self.pseudo_randomize:
            return pseudo_order
    
        rng = np.random.default_rng(self.pseudo_randomize_seed) \
            if self.pseudo_randomize_seed is not None else np.random.default_rng()
    
        for i in range(pseudo_order.shape[1]):
            rng.shuffle(pseudo_order[:, i])
    
        return pseudo_order
        
    def _validate_val_scaler(self, val_scaler):
        """
        Normalize strategy-level scaling factors to the strategy count.

        Missing factors are filled with 1, excess factors are discarded, and
        an omitted sequence becomes one factor of 1 per strategy.

        Args:
            val_scaler (sequence, optional): Proposed strategy-level factors.
        """
        # Check if a val_scaler list was provided
        if val_scaler is not None:
            # ``nTrials_cumsum`` has one entry per configured strategy.
            len_diff = len(self.nTrials_cumsum) - len(val_scaler)
            # If lengths are equal, no adjustment is needed
            if len_diff == 0:
                self.val_scaler = val_scaler
            else:
                # Adjust val_scaler based on len_diff
                if len_diff > 0:
                    # Pad missing strategy-level factors with neutral scaling.
                    self.val_scaler = val_scaler + [1] * len_diff
                    # Use warnings.warn to issue a warning
                    warnings.warn(f"The number of val scalers ({len(val_scaler)}) is less"+\
                                  " than the expected number ({len(self.nTrials_cumsum)}).")
                    print(" Padding with 1's to match the count.")
                else:
                    # Ignore factors that do not correspond to a strategy.
                    self.val_scaler = val_scaler[:len(self.nTrials_cumsum)]
                    warnings.warn(f"The number of val scalers ({len(val_scaler)}) is greater"+\
                                  " than the expected number ({len(self.nTrials_cumsum)}).")
                    print("Trimming the list to match the count.")
        else:
            # Set default val_scaler if none provided
            self.val_scaler = [1] * len(self.nTrials_cumsum)
            
    def _init_trial_lists(self, prefix=""):
        """
        Initialize one storage list for every field in the active schema.

        Because the schema is a class attribute, this method automatically
        initializes the additional ``x2``/``U2``/``pX2`` lists when invoked on
        a suprathreshold subclass instance.
    
        Args:
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        for field in self._trial_data_fields:
            setattr(self, f"{prefix}{field}_list", [])        
            
    def _configure_trial(self, client, trial_counter, config_index):
        """
        Configure a condition on its first AEPsych trial, then resume it.
        
        Args:
            client: Client connected to the AEPsych server.
            trial_counter (int): Within-condition AEPsych trial index.
            config_index (int): Condition/configuration index.
        """
    
        if trial_counter == 0:
            client.configure(config_str=self.config_all[config_index],\
                config_name=f"{self.expt_dim}d_colorDiscrimination_idx{config_index}")
        else:
            client.resume(config_name=f"{self.expt_dim}d_colorDiscrimination_idx{config_index}")
    
    def _derive_xref_x1(self, trial_counter, trial_val, config_index = None):
        """
        Convert a trial returned by AEPsych into reference and comparison stimuli
        in the normalized W coordinate system.

        The meaning of `trial_val` depends on `self.expt_dim`:

        - 2D/3D interleaved experiments:
          `trial_val` contains only the comparison offset relative to a fixed
          reference selected by `config_index`. The reference is looked up from
          `self.ref[config_index]`, the comparison offset is optionally scaled,
          and the comparison stimulus is formed as `xref + delta`.

        - 4D/6D full-field experiments:
          `trial_val` contains both the reference coordinates and the comparison
          deltas. The first half of the vector is interpreted as the reference,
          and the second half as comparison offsets from that reference. Only
          the delta portion is scaled.

        - 5D experiments with an ancillary variable:
          `trial_val` contains a 2D reference in the color plane plus a 2D
          comparison delta. The ancillary coordinate is not sampled by AEPsych
          within this method; instead it is supplied separately through
          `self.ref[config_index]` and appended to both stimuli. The comparison
          inherits the same ancillary value, so its ancillary delta is fixed at 0.
        
        Args:
            trial_counter (int): Current trial number to determine the scaling factor.
            trial_val (list): Trial parameters proposed by AEPsych. Their role
                depends on the experiment dimensionality described above.
            config_index (int): Index used to select the fixed reference or
                ancillary slice for interleaved/slice-based simulations. This is
                not needed for full 4D/6D trials where the reference is already
                contained in `trial_val`.
        
        Returns:
            tuple: 
                xref (jax.numpy.array): Reference stimulus in normalized W units.
                x1 (jax.numpy.array): Comparison stimulus in normalized W units.
                trial_val_report (list): Trial values after any delta scaling has
                    been applied, arranged in the same representation expected by
                    downstream logging/reporting code.
        """
    
        if self.expt_dim in [2, 3]:  # Interleaved low-dimensional experiment.
            # Scale the comparison offset, then add it to the fixed reference for
            # the current config/condition.
            trial_val_report = self._apply_val_scaling(trial_counter, trial_val)
            xref = jnp.array(self.ref[config_index])               
            x1 = jnp.array(trial_val_report) + xref
        elif self.expt_dim in [4, 6]:  # Full reference-plus-delta parameterization.
            # Split the sampled vector into reference coordinates and comparison
            # deltas. Only the deltas are rescaled over strategy stages.
            trial_val_ref = trial_val[:self.expt_dim//2]
            trial_val_delta_comp = trial_val[(self.expt_dim//2):]
            trial_val_delta_comp_scaled = self._apply_val_scaling(\
                trial_counter, trial_val_delta_comp)
            # Report the unmodified reference together with the scaled comparison
            # deltas in the original trial layout.
            trial_val_report = trial_val_ref + trial_val_delta_comp_scaled
            xref = jnp.array(trial_val_ref)
            
            # Construct the comparison by applying the scaled delta to the
            # sampled reference.
            x1 = jnp.array(trial_val_delta_comp_scaled) + xref
        else:  # 5D: 2D reference, 2D delta, and one fixed ancillary coordinate.
            # The ancillary coordinate is fixed for this slice and injected from
            # `self.ref`, not sampled as part of `trial_val`.
            ancillary_val = [self.ref[config_index]]
            # The first two values are the reference in the color plane; the
            # remaining two are comparison deltas in that plane.
            trial_val_ref = trial_val[:2]
            trial_val_delta_comp = trial_val[2:]
            trial_val_delta_comp_scaled = self._apply_val_scaling(\
                trial_counter, trial_val_delta_comp)
            # Report only the sampled 4D slice coordinates; the ancillary value
            # is tracked separately through the condition index.
            trial_val_report = trial_val_ref + trial_val_delta_comp_scaled
            xref = jnp.array(trial_val_ref + ancillary_val)
            
            # Apply the color-plane deltas to the reference and keep the
            # ancillary coordinate unchanged by appending a zero delta.
            x1 = jnp.array(trial_val_delta_comp_scaled + [0]) + xref            
        return xref, x1, trial_val_report
    
    def _apply_val_scaling(self, trial_counter, trial_val):
        """
        Scale comparison-offset parameters for the current AEPsych trial.

        A customized trial-level factor takes priority. Otherwise the cumulative
        strategy quotas determine which strategy-level factor applies.
    
        Args:
            trial_counter (int): Within-condition AEPsych trial index.
            trial_val (list): Comparison offsets to scale.

        Returns:
            list: Scaled comparison offsets.
        """
        
        # Determine the scaling factor index based on how many trials have been completed.
        if self.customized_val_scaler is None:
            val_scaler_idx = np.searchsorted(self.nTrials_cumsum, trial_counter, side='right')
            val_scaler_i = self.val_scaler[val_scaler_idx]
        else:
            val_scaler_i = self.customized_val_scaler[trial_counter]
        #initialize
        trial_val_scaled =[]
        
        # In 2D/3D, every proposed parameter is a comparison offset.
        if self.expt_dim == 2 or self.expt_dim == 3:
            len_par = len(self.parnames)
        else:
            # In 4D/5D/6D parameterizations, this helper receives one half of
            # the sampled parameters: the comparison-offset portion.
            len_par = len(self.parnames)//2
            
        for i in range(len_par): 
            # Directly scale each parameter, not relative to any reference.
            trial_val_scaled_i = trial_val[i]*val_scaler_i
            trial_val_scaled.append(trial_val_scaled_i)
        return trial_val_scaled
      
    def _predict_probability_correct(self, xref, x1):
        """
        Simulate a threshold response for one reference/comparison pair.

        The Wishart model produces the local representations used by the oddity
        observer. ``pX1`` is the probability of selecting the comparison, and
        ``binaryResp`` is one Bernoulli draw from that probability.
        
        Args:
            xref (array-like): Normalized dimensions of the reference stimulus,
                               values ranging from -1 to 1.
            x1 (array-like): Normalized dimensions of the comparison stimulus,
                              similar scale as xref.

        Returns:
            tuple: ``(Uref, U1, signed_diff, pX1, binaryResp)``.
        """
        # compute weighted sum of basis function at the reference
        Uref = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, xref)
        # compute weighted sum of basis function at the comparison
        U1   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x1)
        
        # Simulate the decision-making process for identifying the odd stimulus.
        signed_diff = oddity_task.simulate_oddity_one_trial(
            (xref, x1, Uref, U1), self.gt_Wishart.opt_key, 
            self.gt_Wishart.opt_params['mc_samples'],
            self.gt_Wishart.model.diag_term)
        # Convert the signed decision-variable samples into P(select x1).
        pX1 = oddity_task.approx_cdf_one_trial(0.0, signed_diff,
                                               self.gt_Wishart.opt_params['bandwidth'])
        
        # Response randomness is independent of ``pseudo_randomize_seed``.
        randNum = np.random.rand() 
        binaryResp = int(randNum < pX1)
        
        return Uref, U1, signed_diff, pX1, binaryResp
                
    def _update_trial_lists(self, *, prefix="", **trial_data):
        """
        Append one trial's data to the corresponding storage lists.
    
        Fields not supplied are stored as None. This supports experiment
        runs where model-derived values such as Uref are unavailable.

        Args:
            prefix (str): Optional namespace for a parallel set of lists, such
                as ``"pregenerated_"``.
            **trial_data: Field values keyed by names in
                ``self._trial_data_fields``.

        Raises:
            ValueError: If a supplied field is not part of the active schema.
        """
        unknown_fields = set(trial_data) - set(self._trial_data_fields)
    
        if unknown_fields:
            raise ValueError(
                f"Unrecognized trial-data fields: {unknown_fields}"
            )
    
        for field in self._trial_data_fields:
            value = trial_data.get(field)
            getattr(self, f"{prefix}{field}_list").append(value)
                
    def _stack_them_all(self, stacking_ax=0, prefix=""):
        """
        Stack completed trial lists into ``<field>_all`` arrays.

        ``binaryResp`` is converted directly to an array. Other fields are
        stacked only when every stored value is available; fields containing
        ``None`` remain list-only because model-derived values are optional in
        real-participant experiments.
        
        Args:
            stacking_ax (int): Axis along which to stack the data.
            prefix (str): Prefix for the attribute names (e.g., "pregenerated_").
        """
        for field in self._trial_data_fields:
            list_name = f"{prefix}{field}_list"
            array_name = f"{prefix}{field}_all"
            values = getattr(self, list_name)
    
            if field == "binaryResp":
                setattr(self, array_name, jnp.array(values))
            elif len(values) == 0:
                setattr(self, array_name, jnp.array([]))
            elif all(value is not None for value in values):
                setattr(self, array_name, jnp.stack(values, axis=stacking_ax))
    
    def _simulate_pregenerated_trial(self, pregenerated_trials):
        """Simulate and store one threshold pregenerated trial.

        Args:
            pregenerated_trials (dict): Arrays named ``xref`` and ``x1``.
                ``self.pregenerated_trial_counter`` selects the current row.

        Returns:
            int: Simulated binary response.

        Notes:
            The suprathreshold subclass overrides this hook to load ``x2`` and
            apply its three-stimulus response model.
        """
    
        xref = pregenerated_trials["xref"][self.pregenerated_trial_counter]
        x1 = pregenerated_trials["x1"][self.pregenerated_trial_counter]
    
        Uref, U1, signed_diff, pX1, binaryResp = (
            self._predict_probability_correct(xref, x1)
        )
    
        self._update_trial_lists(
            xref=xref,
            x1=x1,
            binaryResp=binaryResp,
            Uref=Uref,
            U1=U1,
            signed_diff=signed_diff,
            pX1=pX1,
            prefix="pregenerated_",
        )
    
        return binaryResp
    
    def _process_trial(self, trial_counter, trial_val, config_index):
        """Process one threshold AEPsych proposal using a simulated response.

        Args:
            trial_counter (int): Within-condition AEPsych trial index.
            trial_val (list): Parameter values returned by ``client.ask()``.
            config_index (int): Condition index used to select fixed values.

        Returns:
            tuple: ``(trial_val_report, trial_data)``. ``trial_val_report`` is
            sent back to AEPsych and ``trial_data`` matches the active storage
            schema.

        Notes:
            This is an override hook. Simulation subclasses can change the
            response model, while experiment subclasses can present the derived
            stimuli and collect a participant response instead.
        """
    
        xref, x1, trial_val_report = self._derive_xref_x1(
            trial_counter,
            trial_val,
            config_index,
        )
    
        Uref, U1, signed_diff, pX1, binaryResp = (
            self._predict_probability_correct(xref, x1)
        )
    
        trial_data = {
            "xref": xref,
            "x1": x1,
            "Uref": Uref,
            "U1": U1,
            "signed_diff": signed_diff,
            "pX1": pX1,
            "binaryResp": binaryResp,
        }
    
        return trial_val_report, trial_data

    def _monitor_time_insert_pregenerated_trials(self, start_time, max_wait_time,
                                                 pregenerated_trials, stop_event):
        """Insert pregenerated trials while an AEPsych ask is still pending.

        The monitor invokes ``_simulate_pregenerated_trial`` polymorphically,
        so the required dictionary fields depend on the concrete simulator.
        It may insert multiple trials if the ask remains pending across
        multiple timeout intervals.

        Args:
            start_time (float): Start time from ``time.time()``.
            max_wait_time (float): Seconds between fallback insertions.
            pregenerated_trials (dict): Pregenerated stimulus arrays.
            stop_event (threading.Event): Set when ``client.ask()`` completes.
        """
        while not stop_event.is_set():  # Exit if stop_event is set
            elapsed_time = time.time() - start_time
            if elapsed_time > max_wait_time:
                print(f"Deadline exceeded ({elapsed_time:.2f}s).")

                # Break the loop if all pre-generated trials are exhausted
                if self.pregenerated_trial_counter >= len(pregenerated_trials['xref']):
                    print("All pre-generated trials have been used.")
                    break
                                    
                # simulated binary response
                print("Running a pre-generated trial...")
                binaryResp = self._simulate_pregenerated_trial(pregenerated_trials)
                print(f"Simulated response (#trial {self.pregenerated_trial_counter}): {binaryResp}")
                
                # Increase trial counter
                self.pregenerated_trial_counter += 1
                
                # Reset the start time for the next pre-generated trial
                start_time = time.time()
    
            time.sleep(0.05)  # Check every 50 ms
       
    def run_simulation(self, client, pregenerated_trials = None, max_wait_time = None):
        """
        Run the shared AEPsych simulation loop for every condition.

        AEPsych proposals are converted into complete trial results through the
        concrete class's ``_process_trial`` hook. When a timeout is
        configured, pregenerated trials are simulated while ``client.ask()``
        remains pending. These fallback trials are inserted dynamically rather
        than according to a predefined MOCS/AEPsych sequence.
    
        Args:
            client (object): AEPsych client instance used to configure and query trials.
            pregenerated_trials (dict, optional): Pregenerated stimulus arrays.
                Threshold simulation requires ``xref`` and ``x1``; the
                suprathreshold subclass additionally requires ``x2``.
            max_wait_time (float, optional): Seconds to wait before each
                pregenerated fallback trial. If omitted, no fallback-monitor
                thread is started.

        Raises:
            ValueError: If ``max_wait_time`` is provided without
                ``pregenerated_trials``.
        """
        # Timeout insertion is optional; when enabled, it uses storage lists
        # matching the concrete class's trial-data schema.
        if max_wait_time is not None:
            # If a time limit is provided, ensure pregenerated_trials are also provided
            if pregenerated_trials is None:
                raise ValueError(
                    "A max_wait_time was specified, but no pregenerated_trials were provided. "
                    "Please supply pregenerated_trials to use during long wait times."
                )
            else:
                self.pregenerated_trial_counter = 0
                self._init_trial_lists(prefix="pregenerated_")     
                flag_insert_pregen = True
        else:
            # If no time limit is provided, pregenerated trials are not used
            flag_insert_pregen = False
    
        trial_counter = 0
        time_elapsed = []
        finished = False
        while not finished:
            print(trial_counter)
            
            for i in range(self.numConfig):
                ii = self.pseudo_order[i, trial_counter]
                
                # The timer includes AEPsych computation and any fallback
                # trials presented while waiting for the ask to complete.
                start_time = time.time()
                
                self._configure_trial(client, trial_counter, ii)
                
                if flag_insert_pregen:
                    # Create an event for stopping the monitoring thread
                    stop_event = threading.Event()
                    # Start monitoring in a separate thread
                    pregenerated_trial_counter_before = self.pregenerated_trial_counter
                    monitor_thread = threading.Thread(target=self._monitor_time_insert_pregenerated_trials,
                                                      args=(start_time, 
                                                            max_wait_time, 
                                                            pregenerated_trials,
                                                            stop_event))  # Pass stop_event
                    monitor_thread.start()
                
                # Request a new trial configuration from the AEPsych client
                trial_AEPsych = client.ask()
                
                if flag_insert_pregen:
                    stop_event.set()  # Signal that `client.ask()` has completed
                    monitor_thread.join()  # Join the monitoring thread to clean up
                #End the timer
                end_time = time.time()
                
                # Preserve the parameter order declared by the AEPsych config.
                trial_val = []
                for s in self.parnames:
                    trial_val.append(trial_AEPsych["config"][s][0])
                
                # Dispatch to the threshold or suprathreshold implementation.
                trial_val_report, trial_data = \
                    self._process_trial(trial_counter, trial_val, ii)
                
                client.tell(
                    config=dict(zip(self.parnames, trial_val_report)),
                    outcome=trial_data["binaryResp"],
                )
                
                self._update_trial_lists(**trial_data)
                
                # Estimate AEPsych-only computation time by subtracting one
                # timeout interval for each fallback trial that was inserted.
                if flag_insert_pregen:
                    pregenerated_trial_counter_after = self.pregenerated_trial_counter
                    used_pregenerated_trial = pregenerated_trial_counter_after - pregenerated_trial_counter_before
                    trial_duration = end_time - start_time - used_pregenerated_trial * max_wait_time
                else:
                    trial_duration = end_time - start_time
                time_elapsed.append(trial_duration)  # Record the time for this trial
                
                # This assumes all condition configs share the same trial quotas;
                # the last response in each sweep controls loop termination.
                finished = trial_AEPsych["is_finished"]
                
            # After all reference stimuli were tested
            trial_counter += 1
        
        # Aggregate all the trial data lists into single arrays
        self._stack_them_all()
        if flag_insert_pregen:
            self._stack_them_all(prefix="pregenerated_")
            
        # Record time
        self.time_elapsed = time_elapsed
        
    def _monitor_time_insert_MOCS_trials(self, start_time, max_wait_time,
                                         trial_sequence, expt_counter, trial_counter, 
                                         stop_event, mocs_triggered):
        """
        Bump up threshold MOCS trials while an AEPsych ask is pending.

        The first insertion uses ``max_wait_time[0]``; subsequent insertions
        use the last entry to account for presentation, response, and intertrial
        delays. Results and sequence status are written directly to
        ``trial_sequence``.

        Args:
            start_time (float): Start time from ``time.time()``.
            max_wait_time (sequence): First and subsequent timeout durations.
            trial_sequence: Threshold sequence containing two-stimulus MOCS data.
            expt_counter (int): Experimental-condition index.
            trial_counter (int): Current global sequence position.
            stop_event (threading.Event): Set when ``client.ask()`` completes.
            mocs_triggered (threading.Event): Set after any MOCS insertion.

        Notes:
            This is a threshold-only legacy hook: it loads only ``xref`` and
            ``x1`` and calls the two-stimulus response model. A suprathreshold
            workflow must override it rather than inherit it unchanged.
        """      
        num_bumped_up_MOCS = 0
        while not stop_event.is_set():  # Exit if stop_event is set
            elapsed_time = time.time() - start_time
            max_wait_time_ii = max_wait_time[0] if num_bumped_up_MOCS == 0 else max_wait_time[-1]
            
            # Insert another available MOCS trial after the applicable timeout.
            if elapsed_time > max_wait_time_ii:
                #find the next available MOCS trial in the list
                print(f"Deadline exceeded ({elapsed_time:.2f}s). Running a pre-generated MOCS trial...")
                trial_replacement_idx_MOCSlist, trial_placement_idx_originallist, trial_placement_id = \
                    trial_sequence.bump_up_one_MOCS_trial(expt_counter, trial_counter) 
    
                # Get the stimulus information
                xref = trial_sequence.pregenerated_MOCS['xref'][trial_replacement_idx_MOCSlist]
                x1 = trial_sequence.pregenerated_MOCS['x1'][trial_replacement_idx_MOCSlist]
    
                # Simulate response based on the reference and comparison stimuli
                *_, binaryResp = self._predict_probability_correct(xref, x1)
                print(f"Simulated responses (#trial {trial_replacement_idx_MOCSlist}): {binaryResp}")
    
                # Store simulated responses
                trial_sequence.update_data_MOCS(trial_replacement_idx_MOCSlist,
                                                xref, x1, binaryResp)
                # ``bump_up_one_MOCS_trial`` returns the new placement in the
                # already-updated sequence, so status uses that placement index.
                trial_sequence.set_trial_status(expt_counter, trial_placement_idx_originallist, "Completed")
                
                # Set the flag to indicate a MOCS trial was run
                mocs_triggered.set() 
    
                # Reset the start time for the next pre-generated trial
                start_time = time.time()
                num_bumped_up_MOCS += 1
                trial_sequence.nBumpUp_MOCS += 1
                
                trial_sequence.final_sequence[expt_counter].append(trial_placement_id)
            time.sleep(0.05)  # Check every 50 ms
        
    def run_simulation_wMOCSinserted(self, client, trial_sequence, max_wait_time=[2.4, 3.6]):
        """
        Run the legacy threshold simulation with scheduled MOCS trials.

        The sequence predetermines AEPsych and two-stimulus MOCS positions. If
        an AEPsych ask exceeds a timeout, a later MOCS trial can be moved forward
        and simulated while the ask remains pending.
    
        Args:
            client (object): AEPsych client instance used to configure and query trials.
            trial_sequence (object): Contains trial sequence information for interleaving
                AEPsych and pre-generated MOCS trials.
            max_wait_time (list): Time limits for AEPsych trial generation. The first 
                missed presentation uses max_wait_time[0]; subsequent misses use 
                max_wait_time[1] (accounting for response and inter-trial interval delays).

        Returns:
            object: The updated ``trial_sequence``.

        Notes:
            This method assumes threshold MOCS entries contain only ``xref`` and
            ``x1``. It is not compatible with
            ``SimulateTrialGivenWishart_suprathres`` without an override.
        """    
        time_elapsed = []  # List to store elapsed time for AEPsych trials
        trial_counter = 0  # Counter to track the current trial number
    
        while trial_counter < trial_sequence.nTrials_total:            
            for expt_idx in range(self.numConfig):  
                
                # Check if the trial is already completed
                current_trial_status = trial_sequence.trial_status[expt_idx][trial_counter]
                if "Completed" in current_trial_status or "Completed_in_time" in current_trial_status:
                    # If already completed, mark as skipped and move to the next trial
                    print(f"Skipping trial {trial_counter} as it is already completed.")
                    new_status = current_trial_status + ["Skipped"]
                    trial_sequence.trial_status[expt_idx][trial_counter] = list(new_status)
                    trial_counter += 1
                    continue
                
                # Retrieve trial identity (e.g., 'MOCS_1', 'AEPsych_1')
                trial_identity = trial_sequence.updated_sequence[expt_idx][trial_counter]
                # Extract trial type ('MOCS' or 'AEPsych') and trial index
                trial_type, trial_idx = trial_identity.split('_')
                trial_idx = int(trial_idx)
                print(f"Trial #{trial_counter}: {trial_identity}")
                
                if trial_type == 'MOCS': 
                    # Retrieve xref and x1 for MOCS trials
                    xref = trial_sequence.pregenerated_MOCS['xref'][trial_idx]
                    x1 = trial_sequence.pregenerated_MOCS['x1'][trial_idx]
                    
                    # Simulate the trial and record the binary response
                    *_, binaryResp = self._predict_probability_correct(xref, x1)
                    # Update the MOCS trial data with the simulated response
                    trial_sequence.update_data_MOCS(trial_idx, xref, x1, binaryResp)
                    # Mark the trial as completed within the time window
                    trial_sequence.set_trial_status(expt_idx, trial_counter, "Completed_in_time")
                    
                elif trial_type == 'AEPsych':
                    # Retrieve the trial order for AEPsych
                    ii = self.pseudo_order[expt_idx, trial_idx]
                    # Configure the trial for the AEPsych client
                    self._configure_trial(client, trial_idx, ii)
                    
                    # Flag to track if a MOCS trial was inserted
                    mocs_triggered = threading.Event()
                    # Start timing the AEPsych trial
                    start_time = time.time()  
                    # Create an event to stop the monitoring thread
                    stop_event = threading.Event()
                    # Start monitoring trial generation in a separate thread
                    monitor_thread = threading.Thread(target=self._monitor_time_insert_MOCS_trials,
                                                      args=(start_time, 
                                                            max_wait_time, 
                                                            trial_sequence,
                                                            expt_idx,
                                                            trial_counter,
                                                            stop_event,
                                                            mocs_triggered))  
                    monitor_thread.start()
                
                    # Request a new trial configuration from AEPsych
                    trial_AEPsych = client.ask()
                    
                    # Once AEPsych finishes, stop the monitoring thread
                    stop_event.set()
                    monitor_thread.join()  # Ensure the thread finishes before continuing
                    end_time = time.time()
                    
                    # Extract stimulus dimensions for the trial
                    trial_val = [trial_AEPsych["config"][s][0] for s in self.parnames]
                    # Simulate the trial
                    xref, x1, trial_val_report = self._derive_xref_x1(trial_idx, trial_val, ii)
                    *_, binaryResp = self._predict_probability_correct(xref, x1)
                
                    # Report the result back to AEPsych
                    client.tell(config=dict(zip(self.parnames, trial_val_report)),
                                outcome=binaryResp)
                                
                    # Update trial-related data
                    self._update_trial_lists(xref=xref, x1=x1, binaryResp=binaryResp)
                    
                    # Record elapsed time for the trial
                    trial_duration = end_time - start_time
                    time_elapsed.append(trial_duration)
                    
                    # Mark trial as completed with appropriate status
                    if mocs_triggered.is_set():
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        f"Elapsed_time_{trial_duration:.4f}")
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        "Completed")
                    else:
                        trial_sequence.set_trial_status(expt_idx, trial_counter, 
                                                        "Completed_in_time")
                
                # Record the actual trial sequence that was executed
                trial_sequence.final_sequence[expt_idx].append(trial_identity)
                # Increment trial counter
                trial_counter += 1
      
        # Aggregate trial data lists into single arrays
        self._stack_them_all()
    
        # Record the total elapsed times for AEPsych trials only
        self.time_elapsed = time_elapsed
    
        # Return the updated trial_sequence with new data
        return trial_sequence

#%%
class SimulateTrialGivenWishart_suprathres(SimulateTrialGivenWishart):
    """Simulate three-stimulus suprathreshold comparison trials.

    A response of 1 means that comparison 2 (``x2``) was judged more
    different from ``xref`` than comparison 1 (``x1``). The subclass reuses
    the base simulation loop and storage machinery while overriding the
    stimulus derivation and response-simulation hooks.
    """

    # Replace the threshold schema with its three-stimulus counterpart.
    _trial_data_fields = (
        "xref",
        "x1",
        "x2",
        "Uref",
        "U1",
        "U2",
        "signed_diff",
        "pX2",
        "binaryResp",
    )
    
    def __init__(self, expt_dim, config_all, gt_Wishart, ref = None, 
                 pseudo_randomize = False, pseudo_randomize_seed = None, 
                 val_scaler = None, customized_val_scaler = None,
                 comp1 = None, nTrials_total = None):
        """Initialize a suprathreshold simulator.

        Shared arguments follow ``SimulateTrialGivenWishart``.

        Args:
            expt_dim (int): Supported suprathreshold parameterization: 2, 3,
                4, or 6. A 5D suprathreshold mapping is not implemented.
            ref (sequence): One fixed reference stimulus per condition. Unlike
                the threshold base class, this is used in both the 2D/3D and
                4D/6D suprathreshold mappings.
            comp1 (sequence, optional): Fixed comparison-1 stimuli for 2D/3D
                experiments. In 4D/6D, comparison 1 is instead constructed
                from the first half of each AEPsych proposal.
            nTrials_total (int): Number of positions per lane in the complete
                interleaved sequence. This is required to construct
                ``pseudo_order`` during base-class initialization.

        Raises:
            ValueError: If ``comp1`` is missing for a 2D/3D experiment.
        """
        # These fields must exist before ``super().__init__`` because the base
        # constructor dispatches to this class's pseudorandom-order hook.
        self.comp1 = comp1
        self.nTrials_total = nTrials_total
        
        # Call base-class init with the shared arguments
        super().__init__(expt_dim, config_all, gt_Wishart,
                         ref=ref,
                         pseudo_randomize=pseudo_randomize,
                         pseudo_randomize_seed=pseudo_randomize_seed,
                         val_scaler=val_scaler,
                         customized_val_scaler=customized_val_scaler)
    
        # For 2D or 3D experiments, a fixed comp1 must be provided
        if self.expt_dim in (2, 3) and self.comp1 is None:
            raise ValueError(
                "comp1 must be provided for 2D or 3D suprathreshold experiments."
            )
       
    def _create_pseudorandom_order(self):
        """
        Create condition assignments for every interleaved lane position.

        Each column contains every condition index exactly once. Columns stay
        sequential when ``pseudo_randomize`` is False and are independently
        shuffled otherwise.
    
        Returns:
            np.ndarray: Integer array with shape
                ``(numConfig, nTrials_total)``.
        """
        order_temp = np.array(list(range(self.numConfig)))
        pseudo_order = np.tile(order_temp[:, np.newaxis], (1, self.nTrials_total))
        if not self.pseudo_randomize:
            return pseudo_order
    
        rng = np.random.default_rng(self.pseudo_randomize_seed) \
            if self.pseudo_randomize_seed is not None else np.random.default_rng()
    
        for i in range(pseudo_order.shape[1]):
            rng.shuffle(pseudo_order[:, i])
    
        return pseudo_order
        
    def _derive_xref_x1_x2(self, trial_counter, trial_val, config_index = None):
        """
        Convert an AEPsych proposal into three suprathreshold stimuli.

        For 2D/3D, ``xref`` and ``x1`` are fixed by ``config_index`` and all
        proposed parameters are the scaled offset used to construct ``x2``.
        For 4D/6D, ``xref`` remains fixed by ``config_index``; the first half
        of the proposal is the offset for ``x1`` and the scaled second half is
        the offset for ``x2``.
        
        Args:
            trial_counter (int): Within-condition AEPsych trial index used for
                value scaling.
            trial_val (list): Parameter values returned by ``client.ask()``.
            config_index (int): Condition index selecting the fixed reference
                for every supported dimensionality and fixed ``comp1`` for
                2D/3D.
        
        Returns:
            tuple: 
                xref (jax.numpy.array): Normalized reference stimulus 
                    (W unit: between -1 and 1).
                x1 (jax.numpy.array): Normalized comparison stimulus #1 
                    (W unit: between -1 and 1).
                x2 (jax.numpy.array): Normalized comparison stimulus #2
                    (W unit: between -1 and 1).
                trial_val_report (list): Proposal values after comparison-2
                    scaling, in the order expected by AEPsych ``tell``.
        """
        
        if self.expt_dim == 2 or self.expt_dim == 3:
            # Both anchors are fixed for the condition; AEPsych varies only x2.
            xref = jnp.array(self.ref[config_index])    
            x1 = jnp.array(self.comp1[config_index])
            trial_val_delta_comp2_scaled = self._apply_val_scaling(trial_counter, trial_val)
            x2 = jnp.array(trial_val_delta_comp2_scaled) + xref
            trial_val_report= list(trial_val_delta_comp2_scaled)
            
        elif self.expt_dim == 4 or self.expt_dim == 6:
            # The physical stimulus has half as many coordinates as the
            # AEPsych parameterization, which contains two comparison offsets.
            xref = jnp.array(self.ref[config_index])    
            
            # The first half specifies x1's offset; the second specifies x2's.
            trial_val_delta_comp1 = trial_val[:self.expt_dim//2]
            trial_val_delta_comp2 = trial_val[(self.expt_dim//2):]
            trial_val_delta_comp2_scaled = self._apply_val_scaling(\
                trial_counter, trial_val_delta_comp2)
            # Tell AEPsych the unscaled x1 offset and the scaled x2 offset.
            trial_val_report = trial_val_delta_comp1 + trial_val_delta_comp2_scaled
            
            # Add both offsets to the condition's fixed reference.
            x1 = jnp.array(trial_val_delta_comp1) + xref
            x2 = jnp.array(trial_val_delta_comp2_scaled) + xref
        else:
            # Unlike the base threshold simulator, no 5D suprathreshold
            # parameterization is currently implemented.
            print('not supported right now!')
        return xref, x1, x2, trial_val_report
    
    def _predict_probability_correct(self, xref, x1, x2):
        """
        Simulate one suprathreshold comparison response.

        ``pX2`` is the probability that comparison 2 is judged more different
        from the reference than comparison 1. ``binaryResp`` is one Bernoulli
        draw from that probability.
        
        Args:
            xref (array-like): Normalized dimensions of the reference stimulus,
                               values ranging from -1 to 1.
            x1 (array-like): Normalized dimensions of the comparison stimulus,
                              similar scale as xref.
            x2 (array-like): Normalized dimensions of comparison stimulus 2.

        Returns:
            tuple: ``(Uref, U1, U2, signed_diff, pX2, binaryResp)``.
        """
        
        # compute weighted sum of basis function at the reference
        Uref = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, xref)
        # Compute the model representation for both comparisons.
        U1   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x1)
        U2   = self.gt_Wishart.model.compute_U(self.gt_Wishart.W_est, x2)
        
        # Simulate the decision-making process for identifying the odd stimulus.
        signed_diff = oddity_task.simulate_oddity_suprathres_one_trial(
            (xref, x1, x2, Uref, U1, U2), self.gt_Wishart.opt_key, 
            self.gt_Wishart.opt_params['mc_samples'],
            self.gt_Wishart.model.diag_term)
        # Convert the signed decision-variable samples into P(select x2).
        pX2 = oddity_task.approx_cdf_one_trial(0.0, signed_diff,
                                               self.gt_Wishart.opt_params['bandwidth'])
        
        # Response randomness is independent of ``pseudo_randomize_seed``.
        randNum = np.random.rand() 
        binaryResp = int(randNum < pX2)
        
        return Uref, U1, U2, signed_diff, pX2, binaryResp
    
    def _simulate_pregenerated_trial(self, pregenerated_trials):
        """Simulate and store one suprathreshold pregenerated trial.

        Args:
            pregenerated_trials (dict): Arrays named ``xref``, ``x1``, and
                ``x2``. ``self.pregenerated_trial_counter`` selects the row.

        Returns:
            int: Simulated binary response, where 1 selects comparison 2.
        """
    
        xref = pregenerated_trials["xref"][self.pregenerated_trial_counter]
        x1 = pregenerated_trials["x1"][self.pregenerated_trial_counter]
        x2 = pregenerated_trials["x2"][self.pregenerated_trial_counter]
    
        Uref, U1, U2, signed_diff, pX2, binaryResp = (
            self._predict_probability_correct(xref, x1, x2)
        )
    
        self._update_trial_lists(
            xref=xref,
            x1=x1,
            x2=x2,
            binaryResp=binaryResp,
            Uref=Uref,
            U1=U1,
            U2=U2,
            signed_diff=signed_diff,
            pX2=pX2,
            prefix="pregenerated_",
        )
    
        return binaryResp
        
    def _process_trial(self, trial_counter, trial_val, config_index):
        """Process one suprathreshold proposal using a simulated response.

        Args:
            trial_counter (int): Within-condition AEPsych trial index.
            trial_val (list): Parameter values returned by ``client.ask()``.
            config_index (int): Condition index used to select fixed stimuli.

        Returns:
            tuple: ``(trial_val_report, trial_data)``. ``trial_data`` follows
            the subclass's three-stimulus storage schema.
        """
    
        xref, x1, x2, trial_val_report = self._derive_xref_x1_x2(
            trial_counter,
            trial_val,
            config_index,
        )
    
        Uref, U1, U2, signed_diff, pX2, binaryResp = (
            self._predict_probability_correct(xref, x1, x2)
        )
    
        trial_data = {
            "xref": xref,
            "x1": x1,
            "x2": x2,
            "Uref": Uref,
            "U1": U1,
            "U2": U2,
            "signed_diff": signed_diff,
            "pX2": pX2,
            "binaryResp": binaryResp,
        }

        return trial_val_report, trial_data
        
    
