"""Dummy feature generatr for TSP example."""

import os
import json
import numpy


class TSPFeats():
    """Dummy TSP feature generator."""

    def __init__(self, path: str) -> None:
        """Initialize dummy TSP feature generator..

        :param path: Path to feature files directory.
        :type path: str
        """
        self.features = {}
    
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith(".json"): 
                file_path = os.path.join(path, filename)
                with open(file_path, 'r') as handle:
                    features = list(json.loads(handle.read()).values())
                self.features[filename] = features

    def get_features(self, instance) -> list:
        """Get features for instance.

        :param instance: Name of instance.
        :type instance: str
        :returns: List of features for instance.
        :rtype: list
        """
        instance = instance.split('/')[-1]
        feats = next((v for k, v in self.features.items() if instance in k),
                     None)
        for i, f in enumerate(feats):
            if f[0] is None:
                feats[i][0] = 0
        
        return numpy.asarray([item[0] for item in feats]).reshape(1, -1)
