from atlas_ai.system_info import get_system_info


def test_get_system_info_returns_expected_keys():
    info = get_system_info()

    assert "python_version" in info
    assert "platform" in info
    assert "processor" in info
    assert "machine" in info
