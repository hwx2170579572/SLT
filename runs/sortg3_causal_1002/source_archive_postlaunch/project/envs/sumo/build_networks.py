"""Rebuild the checked-in SUMO networks from their plain-XML sources."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _sumo_tool(name: str, explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return str(path)
        raise FileNotFoundError(path)
    found = shutil.which(name)
    if found:
        return found
    common = Path(r"D:\Program Files (x86)\Eclipse\Sumo\bin") / f"{name}.exe"
    if common.is_file():
        return str(common)
    raise FileNotFoundError(f"Could not find {name}; add the SUMO bin directory to PATH")


def build_all(netconvert: str | None = None) -> None:
    binary = _sumo_tool("netconvert", netconvert)
    intersection = ROOT / "networks" / "intersection"
    cross_left_unreg = ROOT / "networks" / "cross_left_unreg"
    double_merge = ROOT / "networks" / "double_merge"
    roundabout = ROOT / "networks" / "roundabout"
    commands = [
        (
            intersection,
            [
                binary,
                "--node-files",
                "intersection.nod.xml",
                "--edge-files",
                "intersection.edg.xml",
                "--output-file",
                "intersection.net.xml",
                "--no-turnarounds",
                "true",
                "--junctions.corner-detail",
                "8",
                "--sidewalks.guess",
                "true",
                "--sidewalks.guess.max-speed",
                "14",
                "--crossings.guess",
                "true",
                "--crossings.guess.speed-threshold",
                "14",
            ],
        ),
        (
            cross_left_unreg,
            [
                binary,
                "--node-files",
                "cross_left_unreg.nod.xml",
                "--edge-files",
                "cross_left_unreg.edg.xml",
                "--output-file",
                "cross_left_unreg.net.xml",
                "--no-turnarounds",
                "true",
                "--junctions.corner-detail",
                "8",
                "--sidewalks.guess",
                "true",
                "--sidewalks.guess.max-speed",
                "14",
                "--crossings.guess",
                "true",
                "--crossings.guess.speed-threshold",
                "14",
            ],
        ),
        (
            double_merge,
            [
                binary,
                "--node-files",
                "double_merge.nod.xml",
                "--edge-files",
                "double_merge.edg.xml",
                "--output-file",
                "double_merge.net.xml",
                "--no-turnarounds",
                "true",
                "--junctions.corner-detail",
                "8",
            ],
        ),
        (
            roundabout,
            [
                binary,
                "--node-files",
                "roundabout.nod.xml",
                "--edge-files",
                "roundabout.edg.xml",
                "--connection-files",
                "roundabout.con.xml",
                "--output-file",
                "roundabout.net.xml",
                "--no-turnarounds",
                "true",
                "--roundabouts.guess",
                "true",
                "--junctions.corner-detail",
                "8",
            ],
        ),
    ]
    for cwd, command in commands:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--netconvert", default=None, help="Explicit netconvert executable")
    args = parser.parse_args()
    build_all(args.netconvert)


if __name__ == "__main__":
    main()
