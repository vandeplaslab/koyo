import pytest
from koyo.json import read_json_data, write_json_data


@pytest.mark.parametrize("indent", (0, 4))
@pytest.mark.parametrize("data", ({"A": 1}, [0, 1, 2], ["A", "B", 3, 4]))
def test_read_write(tmpdir_factory, data, indent):
    path = str(tmpdir_factory.mktemp("json") / "test.json")
    write_json_data(path, data, indent=indent)

    _data = read_json_data(path)
    assert _data == data
