import pytest

from koyo.json import read_json_data, read_json_gzip, write_json_data, write_json_gzip


@pytest.mark.parametrize("indent", (0, 4))
@pytest.mark.parametrize("data", ({"A": 1}, [0, 1, 2], ["A", "B", 3, 4]))
def test_read_write(tmpdir_factory, data, indent):
    path = str(tmpdir_factory.mktemp("json") / "test.json")
    write_json_data(path, data, indent=indent)

    _data = read_json_data(path)
    assert _data == data


@pytest.mark.parametrize("data", ({"A": 1}, [0, 1, 2], ["A", "B", 3, 4]))
def test_read_write_gzip(tmpdir_factory, data):
    path = str(tmpdir_factory.mktemp("json") / "test.json")
    write_json_gzip(path, data)

    _data = read_json_gzip(path)
    assert _data == data


def test_check_gzip_size(tmpdir_factory):
    path_json = str(tmpdir_factory.mktemp("json") / "test.json")
    path_gz = str(tmpdir_factory.mktemp("json") / "test-gz.json")
    # create large dictionary
    data = {str(i): i for i in range(1000)}
    path_json = write_json_data(path_json, data)
    path_gz = write_json_gzip(path_gz, data)
    assert path_json.stat().st_size > path_gz.stat().st_size
