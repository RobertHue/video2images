import pytest

from pipeline.filesystem import clear_directory


@pytest.fixture
def test_dir(tmp_path):
    """
    Fixture to create a temporary directory with files and subdirectories.

    This fixture sets up a directory structure with a file, a subdirectory,
    and another file within the subdirectory. The temporary directory is
    automatically cleaned up after the test function finishes.

    :param tmp_path: Pytest's temporary path fixture that provides a
                     unique temporary directory.
    :return: Path object of the created test directory.
    """
    d = tmp_path / "test_dir"
    d.mkdir()
    (d / "file1.txt").write_text("content")
    (d / "subdir").mkdir()
    (d / "subdir" / "file2.txt").write_text("content")
    return d

def test_clear_directory(test_dir):
    """
    Test the clear_directory function.

    This test verifies that the clear_directory function correctly
    removes all files and subdirectories within the specified directory.

    :param test_dir: The temporary directory created by the test_dir fixture.
    """
    clear_directory(test_dir)
    assert not any(test_dir.iterdir()), "Directory should be empty after clearing"
