from dataclasses import dataclass

import numpy as np
from numpy import bool_, float64
from numpy.typing import NDArray


def fit_line_tls(P: NDArray[float64]) -> NDArray[float64]:
    # ordinary least squares isn't correct choice. it can break down when fitting
    # vetrical lines. plus, we're interested in minimizing the errors normal to the
    # fit line, not vertical from it. so we use total least squares and solve with svd

    # in a total least squares regression, the fitted line passes through the
    # centroid of the data, where the centroid is the the point represented by
    # the means of each coordinate.
    c = P.mean(axis=0)

    # compute the centered points P - c, which places centroid at origin. use svd to find the
    # line and the normal n. given that the dot product between a vector v and a unit vector n
    # gives the component of the v along that unit vector, we only need to return n to reconstruct
    # the distances of individual points to the fitted line. however, we must also subtract the
    # centroid c from any individual data point p to find the error.
    # this yields (p - c)^T n = p^Tn - c^Tn. thus if we report the normal vector and
    # the adjustment -c^Tn, we can construct the distance simply.
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]  # unit normal: direction of least spread
    return np.array([n[0], n[1], -c @ n])


@dataclass
class SideReport:
    """quality of each side's fit, for weighting or diagnostics later. Arrays of length 4."""

    rms: NDArray[float64]  # RMS perpendicular residual of the points used in the final fit, px
    n_used: NDArray[np.int_]  # edge points in the final fit
    n_rejected: NDArray[np.int_]  # valid points removed as outliers
    rejection_skipped: NDArray[
        bool_
    ]  # outliers were flagged but removing them would leave too few points


def fit_sides(
    points: NDArray[float64],
    valid: NDArray[bool_],
    min_points: int = 5,
    outlier_k: float = 3.0,
    min_sigma: float = 0.02,
) -> tuple[NDArray[float64], NDArray[bool_], SideReport]:
    # initialize empty array to store fitted lines (entries are unitialized)
    lines = np.empty((4, 3))

    # create empty report object
    report = SideReport(
        rms=np.zeros(4),
        n_used=np.zeros(4, dtype=int),
        n_rejected=np.zeros(4, dtype=int),
        rejection_skipped=np.zeros(4, dtype=bool),
    )

    # recall valid checks if a crossing point was found by the edge profile
    # and ensures the brightness difference between high and low is above some
    # minimum contrast level
    used = valid.copy()

    for k in range(4):
        # if there are not enough valid points found by the edge profile
        # throw value error
        n_valid = int(used[k].sum())
        if n_valid < min_points:
            raise ValueError(f"side {k}: only {used[k].sum()} valid edge points")

        # mask points to only include valid profiles and fit a total least squares
        # line to these edge points.
        line = fit_line_tls(points[k][used[k]])

        # calculate perpendicular distance from each point to the fitted line
        # namely, p^Tn - c^Tn where p is the point, c is the data centroid, and
        # n is the unit vector orthogonal to the fitted line. see fit_line_tls.
        d = points[k] @ line[:2] + line[2]

        # calculate median absolute deviation, scaled to match standard deviation
        # for gaussian residuals, but measurement isn't inflated by outliers themselves
        # see https://en.wikipedia.org/wiki/Median_absolute_deviation
        mad = np.median(np.abs(d[used[k]] - np.median(d[used[k]])))

        # set floor to min_sigma so very tight or idential residuals can't shrink sigma to 0
        sigma = max(1.4826 * mad, min_sigma)

        # only keep points whose residuals are within outlier_k * sigmas from the line
        keep = used[k] & (np.abs(d) <= outlier_k * sigma)
        n_keep = int(keep.sum())

        if n_keep < n_valid:
            if n_keep >= min_points:
                # refit without the outliers
                used[k] = keep
                line = fit_line_tls(points[k][keep])
            else:
                # rejection would leave too few points: keep the first fit, but report it
                report.rejection_skipped[k] = True

        lines[k] = line

        # report quality of the final fit
        d_final = points[k][used[k]] @ line[:2] + line[2]
        report.rms[k] = float(np.sqrt(np.mean(d_final**2)))
        report.n_used[k] = int(used[k].sum())
        report.n_rejected[k] = n_valid - report.n_used[k]

    return lines, used, report


def intersect_sides(lines: NDArray[float64]) -> NDArray[float64]:
    # lines:    [l0, l1, l2, l3]
    # shift:    [l3, l0, l1, l2]
    shift = np.roll(lines, 1, axis=0)
    # given two lines of the form l = (a, b, c) where ax + by + c = 0
    # their cross product gives their intersection in homogeneous coordinates
    p = np.cross(shift, lines)
    return p[:, :2] / p[:, 2:3]


def main() -> None:
    import matplotlib.pyplot as plt

    from frc_vision.edges import edge_profiles, find_crossings
    from frc_vision.geometry import S
    from frc_vision.homography import make_H, project_all
    from frc_vision.render import add_noise, compose_frame, render_tag
    from frc_vision.tag import generate_tag

    rng = np.random.default_rng(2791)
    H_true = make_H(30, 5, 1)
    noisy = add_noise(compose_frame([render_tag(H_true, generate_tag(7))]), sigma=1.0, rng=rng)

    border = np.array([(-S, -S), (S, -S), (S, S), (-S, S)])
    true_corners = project_all(H_true, border)
    rough = true_corners + rng.normal(0.0, 1.0, size=(4, 2))

    samples, profiles, _ = edge_profiles(noisy, rough)
    points, valid = find_crossings(samples, profiles)
    _ = fit_line_tls(points[1])
    lines, used, _ = fit_sides(points, valid)
    fitted = intersect_sides(lines)

    err = np.linalg.norm(fitted - true_corners, axis=1)
    print(f"corner error (px): {err.round(3)}  RMS {np.sqrt((err**2).mean()):.3f}")
    print("rough corner error (px):", np.linalg.norm(rough - true_corners, axis=1).round(3))

    colors = ["tab:red", "tab:green", "tab:cyan", "tab:purple"]
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # edge points and fitted lines on the image
    pad = 20
    xlim = (true_corners[:, 0].min() - pad, true_corners[:, 0].max() + pad)
    ylim = (true_corners[:, 1].max() + pad, true_corners[:, 1].min() - pad)
    ax1.imshow(noisy.pixels, cmap="gray", vmin=0, vmax=255)
    closed = np.vstack((fitted, fitted[:1]))
    ax1.plot(closed[:, 0], closed[:, 1], c="yellow", lw=0.8, label="fitted lines")
    for k in range(4):
        ax1.scatter(points[k][used[k], 0], points[k][used[k], 1], s=8, c=colors[k])
    rejected = valid & ~used
    ax1.scatter(
        points[rejected][:, 0],
        points[rejected][:, 1],
        s=30,
        marker="x",
        c="orange",
        label="rejected outliers",
    )
    ax1.scatter(fitted[:, 0], fitted[:, 1], s=30, marker="x", c="yellow", label="fitted corners")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("edge points (colour = side) and fitted lines")
    ax1.set_xlabel("u (px)")
    ax1.set_ylabel("v (px)")

    # residuals: perpendicular distance of each edge point to its fitted line, along the side
    for k in range(4):
        direction = np.array([-lines[k, 1], lines[k, 0]])  # unit vector along the line
        along = (points[k] - fitted[k]) @ direction  # distance from the side's start corner, px
        dist = points[k] @ lines[k, :2] + lines[k, 2]  # signed perpendicular distance, px
        ax2.plot(
            np.abs(along[used[k]]),
            dist[used[k]],
            "o-",
            ms=3,
            lw=0.6,
            c=colors[k],
            label=f"side {k}",
        )
    ax2.axhline(0, c="gray", lw=0.6)
    ax2.set_box_aspect((ylim[0] - ylim[1]) / (xlim[1] - xlim[0]))
    ax2.legend(fontsize=8)
    ax2.set_title("residuals: edge point distance to fitted line")
    ax2.set_xlabel("position along side (px)")
    ax2.set_ylabel("perpendicular distance (px)")

    # zoom on one corner: rough vs fitted vs true
    k = 0
    cu, cv = true_corners[k]
    r = 4
    ax3.imshow(noisy.pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for side in (k - 1, k):  # the two sides that meet at corner k
        a, b, c = lines[side]
        uu = np.linspace(cu - r, cu + r, 2)
        if abs(b) > abs(a):  # more horizontal: solve for v
            ax3.plot(uu, -(a * uu + c) / b, c=colors[side], lw=1)
        else:  # more vertical: solve for u
            vv = np.linspace(cv - r, cv + r, 2)
            ax3.plot(-(b * vv + c) / a, vv, c=colors[side], lw=1)
    ax3.scatter(*rough[k], s=60, marker="x", c="red", label="rough")
    ax3.scatter(*true_corners[k], s=80, marker="+", c="blue", label="true")
    ax3.scatter(*fitted[k], s=60, marker="x", c="yellow", label=f"fitted ({err[k]:.3f} px off)")
    ax3.set_xlim(cu - r, cu + r)
    ax3.set_ylim(cv + r, cv - r)
    ax3.set_aspect("equal")
    ax3.legend(loc="lower right", fontsize=8)
    ax3.set_title(f"corner {k}, zoomed to ±{r} px")
    ax3.set_xlabel("u (px)")
    ax3.set_ylabel("v (px)")
    plt.show()


if __name__ == "__main__":
    main()
