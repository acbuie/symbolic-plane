"""Functions for finding and manipulating data locations."""
import pathlib


def parse_user_path(file_path: str) -> pathlib.Path:
    """Convert a string to a Path, while also checking that the Path exists.

    Args:
        file_path: A string representing a file path.

    Returns:
        A pathlib.Path instance of the file_path, if it exists.

    Raises:
        ValueError: The supplied path does not exist.
    """
    path = pathlib.Path(file_path).expanduser().resolve()

    if path.exists():
        return path
    # TODO: perhaps log the path here?
    else:
        raise ValueError(f"Supplied path: {path} does not exist.")


def find_geojson(directory: str) -> list[pathlib.Path]:
    """Return a list of geoJSON files in a directory.

    Args:
        directory: The path of the directory, as a string. It will be converted to a
          pathlib.Path instance and validated.

    Returns:
        A list of all geoJSON files in a directory. If there are no files, the list will
        be empty.
    """
    validated_directory = parse_user_path(directory)
    files = [
        file
        for file in validated_directory.iterdir()
        if (file.is_file() and file.suffix == ".geojson")
    ]
    # TODO: log any files found.

    return files


if __name__ == "__main__":
    DATA_FOLDER_PATH = parse_user_path(
        "~/School/Graduate/Projects/Symbolic_Plane_Analysis/line_features/"
    )
    print(find_geojson(DATA_FOLDER_PATH))
