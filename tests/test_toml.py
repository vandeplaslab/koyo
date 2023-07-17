import os

import pytest
import toml
from koyo.toml import read_toml_data, write_toml_data


def test_read_01(tmp_path):
    path = os.path.join(tmp_path, "config.toml")
    test_dict = {"version": 0.1, "data": [1, 2, 3]}
    with open(path, "w") as f_ptr:
        toml.dump(test_dict, f_ptr)

    results = read_toml_data(path)
    assert test_dict == results


def test_read_02(tmp_path):
    path = os.path.join(tmp_path, "config.toml")

    with pytest.raises(OSError):
        _ = read_toml_data(path)


def test_write_write_new(tmp_path):
    path = os.path.join(tmp_path, "config.toml")

    test_dict = {"version": 0.1, "data": [4, 5, 6]}
    write_toml_data(path, test_dict)

    # load data
    with open(path) as f_ptr:
        results = toml.load(f_ptr)
    assert test_dict == results

    # change data - should not be overriden
    test_dict = {"data": [4, 5, 6]}
    write_toml_data(path, test_dict, check_existing=False)

    # load data
    with open(path) as f_ptr:
        results = toml.load(f_ptr)
    assert test_dict == results


def test_write_update_existing(tmp_path):
    path = os.path.join(tmp_path, "config.toml")

    test_dict = {"version": 0.1, "data": [4, 5, 6]}

    write_toml_data(path, test_dict)
    with open(path) as f_ptr:
        results = toml.load(f_ptr)
    assert test_dict == results

    add_data_dict = {"newkey": 42}
    write_toml_data(path, add_data_dict, check_existing=True)
    with open(path) as f_ptr:
        results = toml.load(f_ptr)
    test_dict.update(add_data_dict)
    assert add_data_dict != results
    assert test_dict == results
