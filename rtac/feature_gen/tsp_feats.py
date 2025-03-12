"""Dummy feature generatr for TSP example."""

import os
import json


class TSPFeats():
    """Dummy TSP feature generator."""

    def __init__(self, path: str) -> None:
        """Initialize dummy TSP feature generator..

        :param path: Path to feature files directory.
        :type path: str
        """
        self.features = {}

        path = os.fsencode(path)
    
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename.endswith(".json"): 
                file_path = os.path.join(path, filename)
                with open(file_path) as handle:
                    features = list(json.loads(handle.read()).values())
                self.features[filename] = features

    def get_features(self, instance) -> list:
        """Get features for instance.

        :param instance: Name of instance.
        :type instance: str
        :returns: List of features for instance.
        :rtype: list
        """
        return self.features[instance]
