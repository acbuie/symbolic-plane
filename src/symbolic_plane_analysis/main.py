"""Main module."""


from pathlib import Path

import polars as pl
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from symbolic_plane_analysis import node_analysis
from symbolic_plane_analysis.files import find_geojson

# Ister seems to have a hard time. I think it's coordinate system related
SKIP: list[str] = []
ANGLE_BUFFER = 15


def main() -> None:
    """Script entrypoint."""
    console = Console()

    directory = Path(
        "~/School/Graduate/Projects/Symbolic_Plane_Analysis/geojson/lines/"
    )
    results_dir = Path("~/School/Graduate/Projects/Symbolic_Plane_Analysis/results/")
    complex_results_dir = results_dir / Path("complex/")

    # TODO: Create directories

    geojson_files = find_geojson(directory)
    tasks = len(geojson_files)

    with Progress(
        SpinnerColumn(),
        BarColumn(),
        # TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Script progress.", total=tasks)

        results = []

        for feature in sorted(geojson_files):
            # Format name from file stem
            name = feature.stem.title().replace("_", " ")

            progress.console.log(f"Working {name}")

            if name in SKIP:
                progress.update(task, advance=1, name=name)
                continue

            # Perform node analysis
            progress.console.log("\tNode...")
            node_summary_result, node_analysis_result = node_analysis.do_analysis(
                feature,
                angle_buffer=ANGLE_BUFFER,
            )
            results.append(node_summary_result.collect())

            # TODO: Perform polygon analysis
            progress.console.log("\tCell...")

            # Perform the complex analysis
            progress.console.log("\tComplex...")
            complex_analysis = (
                node_analysis_result.group_by("num_points")
                .count()
                .sort("num_points")
                .collect()
            )
            # TODO:
            # polygon_analysis_result.group_by("num_sides").count().collect()

            complex_analysis.write_csv(
                complex_results_dir / f"{feature.stem}_complex.csv"
            )

            progress.update(task, advance=1, name=name)

    # Compile and save to csv file
    results_df: pl.DataFrame = pl.concat(results)
    results_df.write_csv(results_dir / "results.csv")

    console.log("Results saved to csv")


if __name__ == "__main__":
    main()
