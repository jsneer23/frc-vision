import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import bool_, float64
from numpy.typing import NDArray

S = 0.1651 / 2
C = 0.1651 / 8
TAG_FAMILY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
TAG_ID = 7

g = np.linspace(-S, S, 9)
XX, YY = np.meshgrid(g, g)
lattice = np.column_stack((XX.ravel(), YY.ravel()))


def generate_tag(tag_id: int) -> NDArray[np.bool_]:

    cells = cv2.aruco.generateImageMarker(TAG_FAMILY, tag_id, 8, borderBits=1) > 127
    cells = np.pad(cells, 1, constant_values=True)
    cells = np.rot90(cells, 2)

    return cells


def tag_corner_points(tag_id: int) -> NDArray[float64]:
    corner_matrix = tag_corner_matrix(tag_id)
    return lattice[corner_matrix.ravel()]


def tag_corner_matrix(tag_id: int) -> NDArray[bool_]:
    # generate tag
    cells = generate_tag(tag_id)
    # convert to integer array for addition n = a + b + c + d
    cells = np.asarray(cells, dtype=int)

    # get each cell around a lattice point
    a = cells[:-1, :-1]
    b = cells[1:, :-1]
    c = cells[:-1, 1:]
    d = cells[1:, 1:]

    # sum up the number of black cells
    n = a + b + c + d
    # calculate if the cells diagonally across from one another are the same
    diagonal = (a == d) & (b == c)
    # determine if this is a corner in the image
    corner = (n == 1) | (n == 3) | ((n == 2) & diagonal)

    return corner


def count_corners(tag_id: int) -> int:

    corners = tag_corner_points(tag_id)

    return len(corners)


def plot_tag(tag_id: int) -> None:

    cells = generate_tag(tag_id)
    corners = tag_corner_points(tag_id)

    plt.imshow(cells, cmap="gray", interpolation="nearest", extent=(-5 * C, 5 * C, 5 * C, -5 * C))
    plt.scatter(corners[:, 0], corners[:, 1], s=8, c="red")
    plt.gca().set_aspect("equal")
    plt.show()


def corners_histogram() -> None:
    n_tags = TAG_FAMILY.bytesList.shape[0]
    counts = np.array([count_corners(i) for i in range(n_tags)])

    plt.hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5)
    plt.xlabel("corners per tag")
    plt.ylabel("number of tags")
    plt.show()


def main() -> None:
    corners = tag_corner_points(TAG_ID)
    print(corners)
    plot_tag(TAG_ID)
    corners_histogram()


if __name__ == "__main__":
    main()
