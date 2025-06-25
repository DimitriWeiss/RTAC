# RTAC
Realtime Algorithm Configuration Methods

This Software is a reimplementation of the realtime algorithm configurators (RAC) described in "ReACTR: Realtime Algorithm Configuration through Tournament Rankings", "Pool-Based Realtime Algorithm Configuration" and "Realtime gray-box algorithm configuration using cost-sensitive classification" into a collective RAC suite. It also includes extended options regarding logging, input, e.g. parameter space via PCS files, and target algorithm calls.

# Installation

You can use RTAC from the files the github repository. In the local directory run 

```
pip install -e .
```

in the root directory. You can then use the code with

```
python3 -m rtac.main
```

from root directory where main is a python file as described in the Examples Section. You can also install it as a Python package via 
```
pip install rtac
```

After installing, you can test functionality of the library by running

```
from rtac.examples.main import run_example
run_example()
```

in Python. It will make sure that python_tsp 0.4.1 is installed and than configure a python_tsp solver with ReACTR on 98 TSP instances that come with the package. You can call the configurators as described in the Examples Section.