from .base_terrain_wrapper import BaseTerrainWrapper

from .paper_terrains import (
    make_flat, make_pyramid_stairs, make_discrete_obstacles,
    make_wave_terrain, make_sloped_terrain,
    make_rma_rough, make_rma_stepping_stones,
    make_dreamwaq_rough, make_dreamwaq_mixed,
    make_walk_these_ways_rough, make_walk_these_ways_stairs,
    make_anymal_rough, make_anymal_steps,
    make_parkour_hurdle, make_parkour_gap, make_parkour_wall,
    make_pgtt_progressive,
    make_chrl_mixed, make_chrl_challenge,
)

from .custom_terrains import (
    make_crater_field, make_tunnel_exit, make_sand_dunes,
    make_rubble_field, make_asymmetric_slope,
    make_frozen_lake, make_trench_crossing,
)
