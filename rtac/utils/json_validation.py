"""In this module validation of the configuration space json is performed."""

import jsonschema as js
import json
import os


def validateparams(config_space: dict) -> bool:
    """
    Validate the configuration space definition.

    Parameters
    ----------
    config_space : dict
        Configuration space definition.

    Returns
    -------
    bool
        Whether the configuration space is valid.
    """
    path = os.path.dirname(__file__)
    with open(f'{path}/RTACParamSchema.json', 'r') as f:
        config_schema = json.load(f)
    try:
        js.validate(instance=config_space, schema=config_schema)
    except js.exceptions.ValidationError as e:
        print(e)
        return False
    return True


if __name__ == "__main__":
    pass
