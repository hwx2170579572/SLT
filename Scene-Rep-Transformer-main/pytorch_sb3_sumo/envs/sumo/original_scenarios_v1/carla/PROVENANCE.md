# CARLA Town-10 task reconstructed for SUMO

Unlike the five SMARTS scenarios beside this directory, CARLA did not ship a
SUMO network in the authors' release.  This compact SUMO network is newly
constructed from the released `envs/carla/carla_env.py`, `map/wp.npy`, and
`map/wp2.npy` without importing or installing CARLA.

The source task starts at `(0, -64.5)`, follows a shared curved approach, then
offers two northbound lane centers near `x=-48.8` and `x=-52.3`.  Success is
only possible in the latter lane near `(-52.5, -32)`.  Accordingly, the SUMO
route begins in lane 0 and `goal_out` is connected only from lane 1, making a
post-turn lane change mandatory.  East-west traffic and pedestrians cross the
northbound segment at the source coordinates' interaction region.
