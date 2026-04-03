from pathlib import Path

from zakotfs.lattice import derive_support_geometry
from zakotfs.params import load_config


def test_support_geometry_matches_paper():
    cfg = load_config(Path("configs/system.yaml"))
    geom = derive_support_geometry(cfg)
    assert geom.delta_k == 27
    assert geom.delta_l == 43
    assert (geom.k_min, geom.k_max) == (-13, 13)
    assert (geom.l_min, geom.l_max) == (-21, 21)
