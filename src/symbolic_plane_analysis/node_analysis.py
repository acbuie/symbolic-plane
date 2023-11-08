"""Perform the geometric analysis of the line features."""
import pathlib

import numpy.typing as npt
import polars as pl
import shapely

from symbolic_plane_analysis.geometry import (
    calculate_node_angles,
    clip_lines_to_points,
    load_geojson,
)


def create_lines_dataframe(
    linestring_array: npt.NDArray[shapely.LineString],
) -> pl.LazyFrame:
    """Create the polars DataFrame containing the line features.

    Args:
        linestring_array: A numpy array of LineString containing all the line features.

    Returns:
        A polars DataFrame with the line features.
    """
    # Convert linestring to two arrays of coordinates
    point_1 = shapely.get_coordinates(shapely.get_point(linestring_array, 0))
    point_2 = shapely.get_coordinates(shapely.get_point(linestring_array, 1))

    df_linestring = pl.from_numpy(linestring_array, schema=["geometry"], orient="row")
    df_point_1 = pl.from_numpy(point_1, schema=["point_1_x", "point_1_y"], orient="row")
    df_point_2 = pl.from_numpy(point_2, schema=["point_2_x", "point_2_y"], orient="row")

    df = pl.concat([df_linestring, df_point_1, df_point_2], how="horizontal")

    query = df.lazy().select(
        pl.col("geometry"),
        pl.concat_list("point_1_x", "point_1_y").alias("point_1"),
        pl.concat_list("point_2_x", "point_2_y").alias("point_2"),
    )
    return query


def create_nodes_dataframe(
    linestring_array: npt.NDArray[shapely.LineString],
) -> pl.LazyFrame:
    """Create the polars DataFrame containing the valid intersection nodes.

    Args:
        linestring_array: A numpy LineString array containing all the coordinates in the
        linestring.

    Returns:
        A polars DataFrame with valid intersection nodes. Each point in this DataFrame
        is contained in at least 3 lines.
    """
    coords = shapely.get_coordinates(linestring_array)
    coord_df = pl.from_numpy(coords, schema=["x_coord", "y_coord"], orient="row")
    query = (
        coord_df.lazy()
        .with_columns(pl.count("x_coord").over(["x_coord", "y_coord"]).alias("count"))
        .filter(pl.col("count") > 2)  # only count nodes with 3 or more lines
        .unique()  # drop duplicates
        .with_columns(
            pl.struct(["x_coord", "y_coord"])
            .map_elements(
                lambda struct: shapely.Point(struct["x_coord"], struct["y_coord"])
            )
            .alias("geometry")
        )
    )
    return query


def _create_analysis_dataframe(
    node_dataframe: pl.LazyFrame, line_dataframe: pl.LazyFrame, angle_buffer: float
) -> pl.LazyFrame:
    """Create the analysis dataframe.

    This function joins the line dataframe with the node dataframe, and then computes
    several important variables for the nodes analysis. It computes the intersection
    angles, the node type (X, Y, T, ect), and the node regularity type.

    This is a rather large function. It could be broken up, but I see no reason to do
    so. Polars was designed with method chaining in mind, and the lazy evaluation means
    this would likely take longer if the dataframe was broken up into smaller parts.

    Args:
        node_dataframe: The dataframe containing valid intersection nodes.
        line_dataframe: The dataframe containing the line features.
        angle_buffer: The size of the buffer around 90 and 180 degrees for classifying
          X, Y, and T junctions. For example, a buffer size of 10 means a valid T
          intersection will have an angle between 170 and 190 degrees, and the other two
          angles within 80 to 100 degrees.

    Returns:
        The node analysis dataframe. Each row is a node containing it's geometry, it's
        intersection angles in degrees, the number of lines entering the node, the node
        type, and the regularity type.
    """
    # Melt lines dataframe to join with nodes
    melt = line_dataframe.melt(
        id_vars="geometry", value_vars=["point_1", "point_2"]
    ).select(
        pl.all().exclude("value"),
        pl.col("value").list.get(0).alias("x_coord"),
        pl.col("value").list.get(1).alias("y_coord"),
    )

    # Join nodes and lines, aggregate linestrings into lists
    node_analysis = (
        node_dataframe.join(other=melt, on=["x_coord", "y_coord"], how="left")
        .with_columns(  # Convert all shapely objects to string
            pl.col("geometry_right").map_elements(lambda line: shapely.to_wkt(line)),
        )
        .select(["geometry", "geometry_right"])
        .group_by(pl.col("geometry"))
        .agg(pl.col("geometry_right").alias("lines"))  # Group lines into list
        .with_columns(  # Get buffer geometry as string
            pl.col("geometry")
            .map_elements(
                lambda point: shapely.to_wkt(shapely.buffer(point, 1, quad_segs=16))
            )
            .alias("polygon")
        )
        .select(["geometry", "lines", "polygon"])
        .with_columns(  # Calculate node angles
            pl.struct(["polygon", "lines"])  # Grab geometry columns, as strings
            .map_elements(
                lambda struct: calculate_node_angles(
                    struct["polygon"], struct["lines"]
                ),
                return_dtype=pl.List(pl.Float64),
            )
            .list.eval(pl.element().round(2), parallel=True)
            .alias("degrees")
        )
        .with_columns(pl.col("degrees").list.lengths().alias("count"))
        .with_columns(  # Classify each node type as X, T, Y, or #
            pl.when(  # X junction
                (pl.col("count") == 4)
                & (
                    pl.col("degrees")
                    .list.eval(
                        pl.element().is_between(90 - angle_buffer, 90 + angle_buffer),
                        parallel=True,
                    )
                    .list.all()  # need to reenter the list namespace to flatten
                )
            )
            .then(pl.lit("X", dtype=pl.Utf8))
            .when(  # T junction
                (pl.col("count") == 3)
                & (
                    pl.col("degrees")
                    .list.eval(
                        pl.element().is_between(90 - angle_buffer, 90 + angle_buffer)
                        | pl.element().is_between(
                            180 - angle_buffer, 180 + angle_buffer
                        ),
                        parallel=True,
                    )
                    .list.all()
                )
            )
            .then(pl.lit("T", dtype=pl.Utf8))
            .when(  # Y junction
                (pl.col("count") == 3)
                & ~(  # Simply negate the T junction calculation
                    pl.col("degrees")
                    .list.eval(
                        pl.element().is_between(90 - angle_buffer, 90 + angle_buffer)
                        | pl.element().is_between(
                            180 - angle_buffer, 180 + angle_buffer
                        ),
                        parallel=True,
                    )
                    .list.all()
                )
            )
            .then(pl.lit("Y", dtype=pl.Utf8))
            .otherwise(pl.lit("#", dtype=pl.Utf8))  # None, # or other junction
            .alias("node_type")
        )
        .with_columns(  # Classify each node as regular or irregular
            pl.when(pl.col("node_type") == "T")  # T are irregular
            .then(pl.lit("irregular", dtype=pl.Utf8))
            .when(  # X and # are regular
                (pl.col("node_type") == "X") | (pl.col("node_type") == "#")
            )
            .then(pl.lit("regular", dtype=pl.Utf8))
            .when(  # Y are regular or irregular
                (pl.col("node_type") == "Y")  # Irregular Y
                & (
                    pl.col("degrees")
                    .list.eval(
                        pl.element().is_between(180 - angle_buffer, 180 + angle_buffer),
                        parallel=True,
                    )
                    .list.any()
                )
            )
            .then(pl.lit("irregular", dtype=pl.Utf8))
            .otherwise(pl.lit("regular", dtype=pl.Utf8))  # Regular Y
            .alias("regularity")
        )
        .with_columns(  # Irregular nodes have "line" running through (180 angle)
            pl.when(pl.col("regularity") == "irregular")
            .then(pl.col("count") - 1)
            .otherwise(pl.col("count"))
            .alias("count")
        )
        .select(["geometry", "degrees", "count", "node_type", "regularity"])
    )

    return node_analysis


def _create_node_summary_row(
    analysis_df: pl.LazyFrame, angle_buffer: float, row_name: str
) -> pl.LazyFrame:
    query = analysis_df.select(
        [
            pl.lit(row_name).alias("terrain"),
            pl.col("count").mean().round(3).alias("n_bar"),
            pl.col("count").std().round(3).alias("n_bar_std"),
            (
                pl.col("node_type").filter(pl.col("node_type") == "T").count()
                / pl.col("node_type").count()
            )
            .round(3)
            .alias("ratio_T"),
            (
                pl.col("node_type").filter(pl.col("node_type") == "Y").count()
                / pl.col("node_type").count()
            )
            .round(3)
            .alias("ratio_Y"),
            (
                pl.col("node_type").filter(pl.col("node_type") == "X").count()
                / pl.col("node_type").count()
            )
            .round(3)
            .alias("ratio_X"),
            (
                pl.col("node_type").filter(pl.col("node_type") == "#").count()
                / pl.col("node_type").count()
            )
            .round(3)
            .alias("ratio_#"),
            (
                pl.col("degrees")
                .explode()
                .is_between(180 - angle_buffer, 180 + angle_buffer)
                .sum()
                / pl.col("degrees")
                .explode()
                .is_between(90 - angle_buffer, 90 + angle_buffer)
                .sum()
            )
            .round(3)
            .alias("ratio_180_90"),
            (
                pl.col("regularity").filter(pl.col("regularity") == "regular").count()
                / pl.col("regularity").count()
            )
            .round(3)
            .alias("regularity"),
        ]
    )

    return query


def do_analysis(feature_path: pathlib.Path, angle_buffer: int) -> pl.LazyFrame:
    """Perform the node analysis.

    Finds valid intersection nodes in a line feature and computes several angle-based
    parameters.

    Args:
        feature_path: The path to the line features.
        angle_buffer: The size of the buffer for classifying node types. For example,
          a buffer of 10 degrees means a node is a T intersection if it contains 3
          angles: 2 within 10 degrees of 90 degrees, and 1 within 10 degrees of 180
          degrees.

    Returns:
        A LazyFrame row containing the computed parameters from the line features.
    """
    # Read geoJSON and get the intersection nodes
    line_features_array = load_geojson(feature_path)
    nodes_df = create_nodes_dataframe(line_features_array)

    # Clip lines to the node buffer. This uses geopandas
    clipped_lines = clip_lines_to_points(
        nodes_df.select("geometry").collect().to_numpy(), line_features_array
    )

    # Create lines dataframe from our shortened lines
    lines_df = create_lines_dataframe(clipped_lines)

    # Perform analysis and summarize
    node_analysis_df = _create_analysis_dataframe(nodes_df, lines_df, angle_buffer)
    summary_df = _create_node_summary_row(
        node_analysis_df, angle_buffer, feature_path.stem
    )

    return summary_df
