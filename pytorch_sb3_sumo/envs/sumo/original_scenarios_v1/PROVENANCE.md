# Original SMARTS scenario assets used directly by SUMO

These assets come from the paper authors' GitHub Release `v1.0.0`:

`https://github.com/georgeliu233/Scene-Rep-Transformer/releases/download/v1.0.0/smarts_scenarios.tar.gz`

Archive SHA-256:

`EDAB55AAB6D1453876B43F319FECD9615B1232597B847E82B403E79B9A10BB49`

Only the original `map.net.xml`, generated `traffic/*.rou.xml`, and a renamed
copy of `scenario.py` are retained here.  SUMO reads the network and traffic
files directly.  `scenario_source.txt` is evidence for mission/flow settings
and is never imported, so SMARTS is not an installation or runtime dependency.

The `ego.rou.xml` files were newly added for this migration.  Their route,
lane, position, and departure time are transcribed from each original
`gen_missions(...)` call.

## SMARTS observation contract

The compatible release named by the authors is SMARTS v0.4.17. Its official
source establishes that `Waypoints` use the project-default 1 metre spacing,
`NeighborhoodVehicles(radius=None)` applies no distance filter, and
`Heading.from_sumo` uses north as zero with counter-clockwise positive angles.
The source archive inspected for these constants was:

- `https://github.com/huawei-noah/SMARTS/archive/refs/tags/v0.4.17.zip`
- SHA-256 `C2DF95103278409E3F43738A6A34F03C04DD37294389DBFFB8FBF1EA26BB76DC`

Only the source archive was inspected; SMARTS was not installed or imported.
