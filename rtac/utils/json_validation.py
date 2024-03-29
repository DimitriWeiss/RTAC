"""In this module validation of the configuration space json is performed."""

import jsonschema as js
import json
import os


def validateparams(config_space: dict) -> bool:
    """Validate the configuration space definition.

    :param config_space: Configuration space definition.
    :type config_space: dict
    :param logs: Object containing loggers and logging functions.
    :type: RTACLogs
    :returns: If configuration space is valid.
    :rtype: bool
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
