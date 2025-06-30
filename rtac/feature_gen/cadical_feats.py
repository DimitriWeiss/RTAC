"""Dummy feature generator for Cadical example."""

import numpy
import csv
import os


class CadFeats():
    """
    Dummy CNF feature generator.

    Parameters
    ----------
    path : str
        Path to feature files directory.
    """

    def __init__(self, path: str):
        """
        Initialize dummy TSP feature generator.
        """
        self.features = {}

        with open(f'{path}', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            _ = next(reader)
            for row in reader:
                instance_path = row[0]
                instance_name = os.path.basename(instance_path)
                features = [float(x) for x in row[1:]]
                self.features[instance_name] = features

    def get_features(self, instance) -> list:
        """Get features for instance.

        Parameters
        ----------
        instance : str
            Path to the problem instance to get the features for.

        Returns
        -------
        list
            A list of the problem instance features.
        """
        instance = instance.split('/')[-1]

        return numpy.asarray(self.features[instance]).reshape(1, -1)
