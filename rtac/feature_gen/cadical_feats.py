"""Dummy feature generator for Cadical example."""

import numpy
import csv
import os


class CadFeats():
    """Dummy TSP feature generator."""

    def __init__(self, path: str) -> None:
        """Initialize dummy TSP feature generator..

        :param path: Path to feature files directory.
        :type path: str
        """
        self.features = {}

        with open(f'{path}', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            _ = next(reader)
            for row in reader:
                instance_path = row[0]
                instance_name = os.path.basename(instance_path)  # get just the filename like '0539.cnf'
                features = [float(x) for x in row[1:]]
                self.features[instance_name] = features

    def get_features(self, instance) -> list:
        """Get features for instance.

        :param instance: Name of instance.
        :type instance: str
        :returns: List of features for instance.
        :rtype: list
        """
        instance = instance.split('/')[-1]

        return numpy.asarray(self.features[instance]).reshape(1, -1)
