"""Geometry module for handling input geometry.

Contains functions for parsing, altering, and creating geometry based on the line input
features. There are some geospatial operations, but they are coordinate system agnostic.
"""

import pathlib

import geopandas as gpd
import numpy.typing as npt
import shapely


def load_geojson(file: pathlib.Path) -> npt.NDArray[shapely.LineString]:
    """Read a geoJSON file and convert it to a numpy array.

    Assumes the geoJSON only contains line features as LineStrings.

    Args:
        file: The geoJSON, as a pathlib.Path, to convert.

    Returns:
        A numpy array of shapely LineString's as a representation of the JSON.
    """
    with open(file) as f:
        file_contents = f.read()
        geo_collection = shapely.from_geojson(file_contents)
        linestring_array: npt.NDArray[shapely.LineString] = shapely.get_parts(
            geo_collection
        )
        return linestring_array


def clip_lines_to_points(
    point_array: npt.NDArray[shapely.Point],
    linestring_array: npt.NDArray[shapely.LineString],
    buffer_size: int = 2,
) -> npt.NDArray[shapely.LineString]:
    """Clip the line features to a small buffer around each node.

    Note that the buffer size is arbitrary, but small. It should be larger than the
    buffer used to calculate the angles, to ensure that there is overlap between the
    buffer polygon geometry and the line geometry.

    Args:
        point_array: A numpy array of Point to buffer and clip lines to.
        linestring_array: A numpy array of LineString to clip.
        buffer_size: Unitless size of the buffer around each node. Default 2.

    Returns:
        A numpy array of LineString. Each shares a point with a Node and is the length
        of the buffer size.
    """
    # Create geodataframes from linestring and node arrays, to use overlay
    lines_gdf = gpd.GeoDataFrame(linestring_array, columns=["geometry"])
    buffer_gdf = gpd.GeoDataFrame(
        shapely.buffer(point_array, 2, quad_segs=16), columns=["geometry"]
    )

    # Clip lines with overlay
    cut_lines: npt.NDArray[shapely.LinearString] = lines_gdf.overlay(
        buffer_gdf
    ).to_numpy()
    return cut_lines


def _split_polygon_by_linestrings(
    polygon: shapely.Polygon, linestrings: list[shapely.LineString]
) -> shapely.GeometryCollection:
    """Split a polygon into a collection of polygons with many lines.

    Args:
        polygon: The polygon to split.
        linestrings: The lines with which to split the polygon. Assumes the lines cross
          the polygon boundaries.

    """
    all_lines = [*linestrings, polygon.boundary]

    merged_lines = shapely.ops.linemerge(all_lines)
    border_lines = shapely.ops.unary_union(merged_lines)
    decomposition = shapely.polygonize([border_lines])

    return decomposition


def calculate_node_angles(polygon_wkt: str, lines_wkt: list[str]) -> list[float]:
    """Calculate the angles between an arbitrary number of lines entering a point.

    Args:
        polygon_wkt: A polygon representing the buffer around a point, as a WKT string.
        lines_wkt: A list of lines intersecting a point, as WKT strings.

    Returns:
        A list of angles between each line in degrees.
    """
    buffer_polygon = shapely.from_wkt(polygon_wkt)
    lines = [shapely.from_wkt(line) for line in lines_wkt]

    sub_polygons = _split_polygon_by_linestrings(buffer_polygon, lines)
    degrees = [(slice.area / buffer_polygon.area) * 360 for slice in sub_polygons.geoms]
    return degrees
