.. _installation:

Installation Guide
==================

Requirements
------------

The Python dependencies will be installed with the pip package. It is advised to use a virtual environment with Python 3.10.

Installation
------------

You can use RTAC from the files the github repository. In the local directory run 

.. code-block:: bash

    pip install -e .


in the root directory. You can then use the code with

.. code-block:: bash

    python3 -m rtac.main


from root directory where main is a python file as described in the Examples Section. You can also install it as a Python package via 

.. code-block:: bash

    pip install rtac


After installing, you can test functionality of the library by running

.. code-block:: python

    from rtac.examples.main import run_example
    run_example()


in Python. It will make sure that python_tsp 0.4.1 is installed and than configure a python_tsp solver with ReACTR on 98 TSP instances that come with the package. You can call the RAC methods as described in the Examples Section.

