import sys
import pytest
import os.path
from yaml import FullLoader, load as yaml_load


def get_tests(files=[]):
    select = []
    deselect = []
    for file in files:
        if os.path.exists(file):
            with open(file, 'r') as fh:
                data = yaml_load(fh, Loader=FullLoader)
            if 'select' in data and data['select'] is not None:
                select.extend(data['select'])
            if 'deselect' in data and data['deselect'] is not None:
                deselect.extend(data['deselect'])

    deselect = ['--deselect=' + d for d in deselect]

    return select, deselect


if __name__ == '__main__':
    args = sys.argv[1:]
    files = []

    for arg in args:
        if '.yaml' in arg:
            files.append(arg)

    for file in files:
        args.remove(file)

    select, deselect = get_tests(files)

    pytest_cmd = []
    pytest_cmd.extend(select)
    pytest_cmd.extend(deselect)
    pytest_cmd.extend(args)

    pytest.main(pytest_cmd)
