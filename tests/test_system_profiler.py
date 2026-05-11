from atlas_ai.profiler.system_profiler import get_system_profile


def test_get_system_profile():
    profile = get_system_profile()

    assert "python_version" in profile
    assert "platform" in profile
    assert "processor" in profile
    assert "cpu_count" in profile
    assert "disk_total_gb" in profile
    assert "disk_used_gb" in profile
    assert "disk_free_gb" in profile
