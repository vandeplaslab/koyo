"""Test zipping and unzipping of files."""

from koyo.compression import unzip_directory, zip_directory, zip_files


def test_zip_files(tmp_path):
    """Test zipping and unzipping of files."""
    # Create some test files
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"
    file1.write_text("This is file 1.")
    file2.write_text("This is file 2.")

    # Zip the files
    zip_path = tmp_path / "test.zip"
    zip_files(zip_path, file1, file2)

    # Check that the zip file was created
    assert zip_path.exists()

    unzip_dir = tmp_path / "unzipped"
    unzip_dir.mkdir()
    unzip_directory(zip_path, unzip_dir)

    # Check that the unzipped files exist and have the correct content
    unzipped_file1 = unzip_dir / "file1.txt"
    unzipped_file2 = unzip_dir / "file2.txt"
    assert unzipped_file1.exists()
    assert unzipped_file2.exists()
    assert unzipped_file1.read_text() == "This is file 1."
    assert unzipped_file2.read_text() == "This is file 2."


def test_zip_directory(tmp_path):
    """Test zipping and unzipping of directories."""
    # Create a test directory with some files
    dir_to_zip = tmp_path / "test_dir"
    dir_to_zip.mkdir()
    file1 = dir_to_zip / "file1.txt"
    file2 = dir_to_zip / "file2.txt"
    file1.write_text("This is file 1.")
    file2.write_text("This is file 2.")

    # Zip the directory
    zip_path = tmp_path / "test_dir.zip"
    zip_directory(dir_to_zip, zip_path)

    # Check that the zip file was created
    assert zip_path.exists()

    unzip_dir = tmp_path / "unzipped_dir"
    unzip_dir.mkdir()
    unzip_directory(zip_path, unzip_dir)

    # Check that the unzipped files exist and have the correct content
    unzipped_file1 = unzip_dir / "file1.txt"
    unzipped_file2 = unzip_dir / "file2.txt"
    assert unzipped_file1.exists()
    assert unzipped_file2.exists()
    assert unzipped_file1.read_text() == "This is file 1."
    assert unzipped_file2.read_text() == "This is file 2."
