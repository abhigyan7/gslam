from dataclasses import dataclass

@dataclass
class RasterizerConfig:

    near_plane: float = 0.01
    far_plane: float = 1e10
    antialised: bool = False