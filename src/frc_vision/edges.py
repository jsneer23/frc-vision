from dataclasses import dataclass

import cv2
import numpy as np
from numpy import bool_, float64
from numpy.typing import NDArray

from frc_vision.render import Patch


@dataclass
class EdgeProfiles:
    pts: NDArray[float64]  # (4, n, 2)  profile centres, on the predicted edge
    normals: NDArray[float64]  # (4, 2)     unit inward normal per side
    s: NDArray[float64]  # (S,)       step offsets along the normal, px
    values: NDArray[float64]  # (4, n, S)  brightness


def edge_profiles(
    img: Patch,
    corners: NDArray[float64],
    n_samples: int = 20,
    brightness_profile_px: float = 6.0,
    extra_corner_margin: float = 5.0,
    step: float = 0.25,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:

    if brightness_profile_px > 12.0:
        # TODO: dyanmically adjust this limit
        raise ValueError(
            f"brightness_profile_px={brightness_profile_px} exceeds 12 px; a wider profile can "
            "reach past the black border into interior cells"
        )

    half_profile = brightness_profile_px / 2.0

    starts = corners
    ends = np.roll(corners, -1, axis=0)  # rotate corner indicies to 1, 2, 3, 0

    # generate sample points for brightness profile
    u = np.linspace(0, 1, n_samples)  # generate N equally spaced samples for each line along (0, 1)

    # scale vector u to each line length
    # #general way to scale vector u from a to b is t = a + u * (b − a)
    # we have: (b − a) = (1 − m/L) − m/L = 1 − 2m/L
    # therefore t = m/L + u * (1 − 2m/L)
    L = np.linalg.norm(ends - starts, axis=1)  # length of each tag border line
    m = half_profile + extra_corner_margin  # margin from each corner (px) for profile
    t = (m / L)[:, None] + u[None, :] * (1 - 2 * m / L)[:, None]
    # same formula as above: points = start + t * (end − start)
    pts = starts[:, None, :] + t[:, :, None] * (ends - starts)[:, None, :]

    # generate unit length lines for brightness profile orthogonal to estimated corner lines
    d = ends - starts  # tag border directions
    n = np.column_stack((-d[:, 1], d[:, 0]))  # rotate directions 90°
    n /= np.linalg.norm(n, axis=1, keepdims=True)  # normalize to unit length

    # create brightness sample lines that are -3 to 3 pixels long along the tag borders
    s = np.arange(-half_profile, half_profile + step / 2, step)  # + step/2 includes final pixel
    # samples has 4 axes
    # axis  size
    #  0     4          - which side of the tag (top, right, bottom, left)
    #  1     n_samples  - which sample profile location
    #  2     step       - which step along the edge (measured in 0.25px from -3 to 3)
    #  3	 2	        - which coordinate: 0 = u, 1 = v
    samples = pts[:, :, None, :] + s[None, None, :, None] * n[:, None, None, :]

    # compute brightness profile (i.e. places image pixel value at each point in samples)
    # cv2.remap needs the u and v coordinates separately. it also needs (u0, v0) to be (0,0)
    # so we need to subtract out start location if we're passed a patch not a full image
    map_u = (samples[..., 0] - img.u0).reshape(-1, len(s)).astype(np.float32)
    map_v = (samples[..., 1] - img.v0).reshape(-1, len(s)).astype(np.float32)
    img32 = img.pixels.astype(np.float32)
    prof = cv2.remap(img32, map_u, map_v, cv2.INTER_LINEAR)
    # profiles[k, i, j] is the brightness at the point samples[k, i, j]
    # samples is the (u,v) pixel location of said point
    profiles = prof.reshape(4, n_samples, len(s)).astype(np.float64)

    return samples, profiles, s


def find_crossings(
    samples: NDArray[float64],
    profiles: NDArray[float64],
    n_level: int = 4,
    min_contrast: float = 40.0,
) -> tuple[NDArray[float64], NDArray[bool_]]:

    # compute light and dark levels from the ends of each profile
    light = profiles[..., :n_level].mean(axis=-1)
    dark = profiles[..., -n_level:].mean(axis=-1)
    # compute halfway level used to determine edge
    mid = (light + dark) / 2

    # compute first step at or below the midpoint
    below = profiles <= mid[..., None]
    j = np.argmax(below, axis=-1)  # argmax of a boolean array is index of the first True
    found = below.any(axis=-1) & (j > 0)  # bool array for each profile checking midpoint found
    j = np.maximum(j, 1)  # keep j-1 and j valid even where nothing was found

    # find the brightness of the pixels just below and just above the midpoint
    p0 = np.take_along_axis(profiles, (j - 1)[..., None], axis=-1)[..., 0]
    p1 = np.take_along_axis(profiles, j[..., None], axis=-1)[..., 0]

    # calculate the fractional distance between j-1 pixel and j pixel where midpoint falls
    frac = (p0 - mid) / np.where(p0 != p1, p0 - p1, 1.0)  # how far from step j-1 to step j

    # interpolate the (u, v) position of the midpoint
    x0 = np.take_along_axis(samples, (j - 1)[..., None, None], axis=-2)[..., 0, :]
    x1 = np.take_along_axis(samples, j[..., None, None], axis=-2)[..., 0, :]
    points = x0 + frac[..., None] * (x1 - x0)

    valid = found & (light - dark > min_contrast)
    return points, valid


def main() -> None:
    from matplotlib import pyplot as plt

    from frc_vision.geometry import S
    from frc_vision.homography import make_H, project_all
    from frc_vision.render import add_noise, compose_frame, render_tag
    from frc_vision.tag import generate_tag

    rng = np.random.default_rng(2791)
    cells = generate_tag(7)
    H_true = make_H(30, 5, 1)
    patch = render_tag(H_true, cells, F=16)
    frame = compose_frame([patch])
    noisy = add_noise(frame, sigma=1.0, rng=np.random.default_rng())

    border = np.array([(-S, -S), (S, -S), (S, S), (-S, S)])
    true_corners = project_all(H_true, border)
    corners = true_corners + rng.normal(0.0, 1.0, size=(4, 2))

    samples, profiles, s = edge_profiles(noisy, corners)

    points, valid = find_crossings(samples, profiles)

    # plot
    starts_ = samples[:, :, 0, :].reshape(-1, 2)
    ends_ = samples[:, :, -1, :].reshape(-1, 2)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 5))
    pad = 20
    xlim = (true_corners[:, 0].min() - pad, true_corners[:, 0].max() + pad)
    ylim = (true_corners[:, 1].max() + pad, true_corners[:, 1].min() - pad)
    box_aspect = (ylim[0] - ylim[1]) / (xlim[1] - xlim[0])  # height / width of the image view

    ax1.imshow(noisy.pixels, cmap="gray", vmin=0, vmax=255)
    ax1.plot([starts_[:, 0], ends_[:, 0]], [starts_[:, 1], ends_[:, 1]], c="orange", lw=0.8)
    profile_lines = ax1.plot(
        [starts_[:, 0], ends_[:, 0]], [starts_[:, 1], ends_[:, 1]], c="orange", lw=0.8
    )
    profile_lines[0].set_label("profiles")  # 80 lines: label only one so the legend lists it once
    ax1.scatter(starts_[:, 0], starts_[:, 1], s=4, c="yellow", label="profile start (outside)")
    ax1.scatter(corners[:, 0], corners[:, 1], s=30, marker="x", c="red", label="rough corners")
    ax1.scatter(
        true_corners[:, 0], true_corners[:, 1], s=30, marker="+", c="blue", label="true corners"
    )
    pad = 20
    ax1.set_xlim(true_corners[:, 0].min() - pad, true_corners[:, 0].max() + pad)
    ax1.set_ylim(true_corners[:, 1].max() + pad, true_corners[:, 1].min() - pad)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("brightness profiles across each edge")
    ax1.set_xlabel("u (px)")
    ax1.set_ylabel("v (px)")

    ax2.plot(s, profiles[0, :5].T, label=[f"profile {i}" for i in range(5)])
    ax2.legend(fontsize=8)
    ax2.set_box_aspect(box_aspect)
    ax2.set_title("brightness across the edge (side 0, first 5 profiles)")
    ax2.set_xlabel("offset along inward normal (px)")
    ax2.set_ylabel("brightness")

    ax3.imshow(noisy.pixels, cmap="gray", vmin=0, vmax=255)
    closed = np.vstack((true_corners, true_corners[:1]))
    ax3.plot(closed[:, 0], closed[:, 1], c="blue", lw=0.6, label="true edges")
    ax3.scatter(points[valid][:, 0], points[valid][:, 1], s=8, c="lime", label="edge points")
    ax3.scatter(
        points[~valid][:, 0], points[~valid][:, 1], s=20, marker="x", c="red", label="invalid"
    )
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect("equal")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.set_title("measured edge points")
    ax3.set_xlabel("u (px)")
    ax3.set_ylabel("v (px)")
    plt.show()


if __name__ == "__main__":
    main()
