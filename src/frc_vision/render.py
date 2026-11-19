from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from numpy import bool_, float64
from numpy.typing import NDArray

from frc_vision.geometry import N_CELLS, C, S
from frc_vision.homography import make_H, project_all
from frc_vision.tag import generate_tag


@dataclass
class Patch:
    pixels: NDArray[float64]  # (h, w)
    u0: int  # full-image column of pixels[0, 0]
    v0: int  # full-image row of pixels[0, 0]


def render_tag(
    H: NDArray[float64],
    cells: NDArray[bool_],
    image_size: tuple[int, int] = (1280, 800),
    F: int = 16,
    black: float = 30.0,
    white: float = 220.0,
    background: float = 200.0,
    margin: int = 4,
    rows_per_strip: int = 8,
) -> Patch:

    # recall, C is the tag cell size in meters
    # tags are 8 cells wide plus a white border cell on either side
    half = N_CELLS / 2 * C
    border = np.array([(-half, -half), (half, -half), (half, half), (-half, half)])

    # project the border corners onto the image
    uv = project_all(H, border)

    # image width and height in pixels
    width, height = image_size

    # the tag can be askew, so to get a bounding box that contains the projected tag
    # we need to find the minimum and maximum x and y pixel coordinates
    # we also add (or subtract) a margin to account for the blur we add later
    u0 = max(int(np.floor(uv[:, 0].min())) - margin, 0)
    v0 = max(int(np.floor(uv[:, 1].min())) - margin, 0)
    u1 = min(int(np.ceil(uv[:, 0].max())) + margin, width)
    v1 = min(int(np.ceil(uv[:, 1].max())) + margin, height)

    # width and height of the image segment containing the tag
    w, h = u1 - u0, v1 - v0

    # camera pixels measure total light over the entire pixel area. if a line from the tag
    # splits a pixel where 70% is black and 30% is white then the pixel will be gray that
    # is 70% of the way from white to black. in order to determine pixel color we split the
    # pixels into subpixels and assign them white or black depending on color projected to
    # their centers. we then find the color of the actual pixel by averaging the subpixels.

    # start by generating pixel centers for splitting each pixel into F smaller pixels column wise
    offsets = (np.arange(F) + 0.5) / F - 0.5
    # apply center adjustments to pixels and generate list of column coordinates
    us = (np.arange(u0, u1)[:, None] + offsets).ravel()

    # create pixels matrix for the results based on size of boundary box
    pixels = np.empty((h, w))

    # compute H inverse for mapping sub pixel centers to the tag plane (to determine color)
    H_inv = np.linalg.inv(H)

    # compute rows in batches
    for r0 in range(0, h, rows_per_strip):
        # find last row to compute this batch, with maximum h (edge of the boundary box)
        r1 = min(r0 + rows_per_strip, h)

        # row version of pixel center offset calculations
        vs = (np.arange(v0 + r0, v0 + r1)[:, None] + offsets).ravel()

        # layout every combination of u and v out in a gr
        # id
        # np.column_stack((U.ravel(), V.ravel())) would give you a list of all the subpixel points
        # xs = np.array([1, 2, 3])
        # ys = np.array([10, 20])
        # X, Y = np.meshgrid(xs, ys)
        # produces
        # X = [[1, 2, 3],        Y = [[10, 10, 10],
        # [1, 2, 3]]             [20, 20, 20]]
        U, V = np.meshgrid(us, vs)

        # map subpixel centers back to the tag plane
        t = H_inv @ np.stack((U.ravel(), V.ravel(), np.ones(U.size)))
        X = t[0] / t[2]
        Y = t[1] / t[2]

        # recall, tag has 10 cells, which run from -5*C to 5*C in tag frame
        # convert coordinates to cell number in X and Y directions (0 to 9 each)
        ix = np.floor((X + 5 * C) / C).astype(int)
        iy = np.floor((Y + 5 * C) / C).astype(int)

        # boolean mask to eliminate pixels that don't map to the tag
        # t[2] > 0 ensures the projection to the tag plane is in front
        # of the camera (this assumes H is scaled to have positive third component)
        # an estimated H should be scaled with H / H[2,2]
        inside = (ix >= 0) & (ix < 10) & (iy >= 0) & (iy < 10) & (t[2] > 0)

        # create a subpixel array with the background color
        # then assign white or black to the subpixels that are inside the tag
        # notice this produces a flat 1D array of subpixel values
        sub = np.full(U.size, background)
        sub[inside] = np.where(cells[iy[inside], ix[inside]], white, black)

        # reshape the subpixel array into a 4D array with dimensions (rows, F, cols, F)
        # then average over the F dimensions to get the final pixel values
        pixels[r0:r1] = sub.reshape(r1 - r0, F, w, F).mean(axis=(1, 3))

    return Patch(pixels, u0, v0)


def add_noise(patch: Patch, sigma: float, rng: np.random.Generator) -> Patch:
    """Sensor noise, clipped to 8-bit range and rounded like a real camera."""
    noisy = np.clip(np.round(patch.pixels + rng.normal(0.0, sigma, patch.pixels.shape)), 0, 255)
    return Patch(noisy, patch.u0, patch.v0)


def compose_frame(
    patches: list[Patch],
    image_size: tuple[int, int] = (1280, 800),
    background: float = 200.0,
) -> Patch:
    width, height = image_size
    frame = np.full((height, width), background)
    for p in patches:
        h, w = p.pixels.shape
        if p.u0 < 0 or p.v0 < 0 or p.u0 + w > width or p.v0 + h > height:
            raise ValueError(
                f"patch at ({p.u0}, {p.v0}) size {w}x{h} does not fit in {width}x{height}"
            )
        frame[p.v0 : p.v0 + h, p.u0 : p.u0 + w] = p.pixels
    return Patch(frame, 0, 0)


def main() -> None:
    cells = generate_tag(7)
    H_true = make_H(45, 5, 3)
    patch = render_tag(H_true, cells, F=16)
    frame = compose_frame([patch])
    noisy = add_noise(frame, sigma=1.0, rng=np.random.default_rng())
    img = noisy.pixels.astype(np.uint8)

    border = np.array([(-S, -S), (S, -S), (S, S), (-S, S)])

    corners = project_all(H_true, border)

    plt.imshow(cells, cmap="gray", interpolation="nearest", extent=(-S, S, S, -S))

    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.scatter(corners[:, 0], corners[:, 1], s=8, c="red")
    plt.gca().set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    main()
