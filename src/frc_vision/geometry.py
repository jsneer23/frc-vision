"""fixed geometry of an frc 36h11 april tag, in the tag frame (meters, Z = 0, Y down)."""

from typing import Final

import numpy as np
from numpy import bool_, float64
from numpy.typing import NDArray

TAG_SIZE: Final = 6.5 * 2.54 / 100  # convert 6.5 in to meters
N_BLACK: Final = 8  # cells across the black square
N_CELLS: Final = 10  # cells across the full tag incl. white border
C: Final = TAG_SIZE / N_BLACK  # cell size
S: Final = TAG_SIZE / 2  # half the black square

# 9x9 lattice of cell corners over the black square, row by row (row = Y, column = X)
_g = np.linspace(-S, S, N_BLACK + 1)
_xx, _yy = np.meshgrid(_g, _g)
LATTICE: Final[NDArray[float64]] = np.column_stack((_xx.ravel(), _yy.ravel()))
LATTICE.flags.writeable = False

# the black square's 4 outer corners, as a mask over LATTICE
OUTER_CORNER_MASK: Final[NDArray[bool_]] = np.zeros(LATTICE.shape[0], dtype=bool)
OUTER_CORNER_MASK[[0, N_BLACK, N_BLACK * (N_BLACK + 1), -1]] = True
OUTER_CORNER_MASK.flags.writeable = False
