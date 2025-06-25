.. _description:

Description
===========

RTAC: A Realtime Algorithm Configuration Methods Suite
------------------------------------------------------


This Software is a reimplementation of the Realtime Algorithm Configuration (RAC) methods described in "ReACTR: Realtime Algorithm Configuration through Tournament Rankings" :cite:`ReACTR` (ReACTR), "Pool-Based Realtime Algorithm Configuration" :cite:`CPPL` (CPPL), "Realtime gray-box algorithm configuration using cost-sensitive classification" :cite:`GB, GB_extended` (gray-box RAC) and a light extension of ReACTR incorporating additional output at tournament end (ReACTR++) into a collective RAC suite. It also includes options regarding logging, input, e.g. parameter space via PCS files or json, and target algorithm calls. It is possible to use the gray-box functionality with all of the RAC methods implemented in the suite.

Without any warranty or guarantuee, the suite allows for adopting and applying in practical scenarios. Additionally, the library is intended to encourage further research by providing an RAC infrastructure and possibility for easy introduction of further RAC methods. Pull requests at `Github <https://github.com/DimitriWeiss/RTAC>`_ and reporting of bugs are welcome.


RAC
---

.. figure:: _static/figures/Racing_principle.png
   :alt: Realtime Algorithm Configuration through tournaments.
   :width: 500px
   :align: center

   Realtime Algorithm Configuration through tournaments illustrated in runtime minimization scenario.  
   Figure adopted from :cite:`GB`.

In RAC, it is assumed that problem instances arrive in a sequence and are solved without training phase. The current problem instance is to be solved using only information gained from problem instances solved prior to it, optimizing the performance metric (which is taget algorithm runtime in the figure, but can also be solutio quality or RAM consumption of the target algorithm, etc). The problem instance is solved within a tournament of differently configured target algorithms. The performance of the tournament contenders is used to update the assessment method of the configurator which is also used to select contenders for the next tournament, i.e. to solve the next problem instance.


ReACTR
------
ReACTR is an RAC method based on the principle described above. The assessment mechanism is TrueSkill :cite:`Trueskill`, a score-based bayesian skill learning method. It is used to rank configurations within a pool of configurations that is carried along the tournaments. The ranking is used to select contenders for the next tournament and to decide which configurations in the pool are to be replaced by newly generated configurations. New configurations are either generated via genetic crossover or randomly.


ReACTR++
--------
In the ReACTR method, only the difference between the best performning configuration within a tournament and the rest of the configurations as equals, i.e. loosing configurations, is evaluated. This means that the information about configuration quality is censored and the full potential of TrueSkill is not used. ReACTR++ enables a more detailed ranking by providing Trueskill with further information about the configurations state at the end of a tournament. This information could, for example for SAT solvers, be the current number of conflicting clauses or number of original variables left in the formula.

CPPL
----
CPPL replaces TrueSKill by a Contextual Preselection Bandit under the Plackett-Luce. The context is provided by problem instance features. The bandit model is used for selection of contenders, deciding which configurations to replace and for the generation of new configurations in the pool. For this, configurations are first generated through genetic crossover or randomly and then assessed by the bandit model. The configurations deemed to be the best by the bandit model are then inserted into the pool.


Gray-Box RAC
------------

.. figure:: _static/figures/gray_box.png
   :alt: Gray-box Realtime Algorithm Configuration approach.
   :width: 500px
   :align: center

   Gray-box Realtime Algorithm Configuration approach illustrated in runtime minimization scenario. 
   Figure adopted from :cite:`GB_extended`.

The gray-box RAC extension utilizes output gathered from configured target algorithms during problem instance solving runtime. This output gives information about the target algorithms state. It is used to make pairwise comparisons between configurations that that are labeled by which configuration outperformed the other. A cost-sensitive random forest model, implemented using CostCla :cite:`costcla`, is trained and used during tournaments to find underperforming configurations. These runs of the configured target algorithm are terminated and the freed CPU cores are used to prematurely start the next tournament, provided the next problem instance already arrived. In the runtime optimization scenario, the headstart gives an advantage that minimizes the overall runtime over the problem instance sequence (as illustrated in the figure). In the solution quality optimization scenario, the headstart allows for a higher time limit of the early starting configurations which improves the chances of finding a better solution.


References
----------

.. bibliography:: references.bib
   :style: plain