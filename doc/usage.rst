.. _usage:

Usage
=====

Necessary Provisions
--------------------

After you installed rtac, you need to provide data for the AC scenario to be complete.

* Target algorithm.
* A text file containing a defitionion of the scenario.
   * Hyperparameters that you want to adjust.
   * A dotted module path to the target algorithm wrapper.
   * The class name of the wrapper
   * An absolute path to the instance feature generator if you want to use instance features.
   * The class name of the feature generator.
   * An absolute path to a csv file containing instance features if you precomputed them and the instance feature generator only reads them in.
   * A text file with instance features to pre-train CPPL if you use CPPL.
   * An absolute path to a json or pcs file defining the parameter space of the target algorithm.
* Problem instances.
* Absolute paths to the problem instances to be solved.

You can read in the scenario with scenario = rtac.utils.read_io.read_args('path_to_scenario_file'), initialize the RAC with rtac = rtac.rtac.rtac_factory(scenario) and solve instances with rtac.solve_instance(instance).

Hyperparameters
---------------

Below, you will find a table with hyperparameters accepted by the RAC methods in RTAC. You can set them within the scenario file or in the command line.

.. list-table:: **Hyperparameters for RTAC**
   :widths: 30 20 20 80
   :header-rows: 1

   * - Name
     - Type
     - Default
     - Description
   * - `--verbosity`
     - int
     - 0
     - Verbosity level. [0,1,2]
   * - `--number_cores`
     - int
     - 1
     - Number of cores to be used in parallel.
   * - `--wrapper`
     - str
     - No wrapper chosen!
     - Module of Python wrapper for the algorithm.
   * - `--wrapper_name`
     - str
     - No wrapper class name.
     - Name of the wrapper class in the wrapper module.
   * - `--feature_gen_name`
     - str
     - No feature generator class name.
     - Name of the feature generator class in the feature generator module.
   * - `--feature_path`
     - str
     - No feature directory path.
     - Path to the directory with feature files, if existing.
   * - `--timeout`
     - int
     - 300
     - Stop solving single instance after (int) seconds. [300]
   * - `--runtimePAR`
     - int
     - 1
     - Multiply `--timeout` by `--runtimePAR` if instance not solved. [1]
   * - `--contenders`
     - int
     - 30
     - The number of contenders in the pool. [30]
   * - `--keeptop`
     - int
     - 4
     - Number of top contenders automatically in tournament (ReACTR/ReACTR++).
   * - `--chance`
     - int
     - 25
     - Chance to replace gene randomly in ReACTR/ReACTR++. [0–100]
   * - `--mutate`
     - int
     - 10
     - Mutation chance in crossover in ReACTR/ReACTR++. [0–100]
   * - `--kill`
     - float
     - 5
     - Kill contenders with higher variance in ReACTR/ReACTR++.
   * - `--feature_gen`
     - str
     - ""
     - Python wrapper for computing instance features.
   * - `--instance_pre_train`
     - str
     - False
     - Path to pre-training feature file for the bandit.
   * - `--nc_pca_f`
     - int
     - 3
     - Number of PCA dimensions for instance features.
   * - `--nc_pca_p`
     - int
     - 5
     - Number of PCA dimensions for parameter features.
   * - `--jfm`
     - str
     - polynomial
     - Joined feature map mode for CPPL.
   * - `--omega`
     - float
     - 1.0
     - Omega parameter for CPPL.
   * - `--gamma`
     - float
     - 1
     - Gamma parameter for CPPL.
   * - `--alpha`
     - float
     - 0.2
     - Alpha parameter for CPPL.
   * - `--epsilon`
     - float
     - 0.9
     - Epsilon for epsilon-greedy selection (0.0 = greedy).
   * - `--kappa`
     - float
     - 1.0
     - Weight on confidence in pairwise comparison (CPPL).
   * - `--gen_mult`
     - int
     - 2
     - Generation multiplier before CPPL insertion.
   * - `--ac`
     - int
     - 1
     - Algorithm Configuration method [1=ReACTR, 2=ReACTR++, etc.].
   * - `--paramlimit`
     - float
     - 100000
     - Parameter value limit for normalization in CPPL.
   * - `--win_bonus`
     - float
     - 0.2
     - Boost for recent CPPL winner skill.
   * - `--win_decay`
     - float
     - 0.8
     - Decay factor for CPPL skill estimates.
   * - `--recent_winner_boost`
     - float
     - 3.0
     - Absolute skill boost for recent CPPL winner.
   * - `--forgetting_factor`
     - float
     - 0.98
     - Forgetting factor for old observations (theta_bar).
   * - `--obs_noise`
     - float
     - 1.0
     - Assumed observation noise variance in CPPL.
   * - `--cppl_reward`
     - float
     - 1.0
     - Reward for winning arm in CPPL bandit model.
   * - `--cpplt`
     - float
     - 2
     - Reset value for t in CPPL when pool changes.
   * - `--log_folder`
     - str
     - logs
     - Directory for logging.
   * - `--param_file`
     - str
     - No parameter file given!
     - Path to PCS or RTAC JSON parameter file.
   * - `--resume`
     - bool
     - False
     - Resume RTAC from logged configuration state.
   * - `--baselineperf`
     - bool
     - False
     - Run only default parameterizations.
   * - `--pws`
     - bool
     - False
     - Insert default configuration if flag set.
   * - `--objective_min`
     - bool
     - False
     - Optimize for objective value minimization.
   * - `--experimental`
     - bool
     - False
     - Use tournament 0 logs for experiment.
   * - `--online_instance_train`
     - bool
     - False
     - Enable continuous PCA fitting on incoming instances.
   * - `--epsilon_greedy`
     - bool
     - False
     - Enable epsilon greedy selection.
   * - `--isolate_bandit`
     - bool
     - False
     - Run bandit in child process to avoid resource conflict.
   * - `--gray_box`
     - bool
     - False
     - Enable gray-box RAC.
   * - `--gb_read_time`
     - float
     - 0.1
     - Frequency to check gray-box output [seconds].
   * - `--nr_gb_feats`
     - int
     - 2
     - Number of gray-box features used.
