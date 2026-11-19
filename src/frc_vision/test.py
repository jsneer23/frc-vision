import numpy as np
from matplotlib import pyplot as plt
from numpy import bool_, float64
from numpy.typing import NDArray

from frc_vision.geometry import LATTICE, OUTER_CORNER_MASK
from frc_vision.homography import dlt, make_H, project_all
from frc_vision.tag import tag_corner_matrix

# true corners
outer_corners: NDArray[float64] = LATTICE[OUTER_CORNER_MASK]
H_true = make_H(30, 5, 1)
true = project_all(H_true, outer_corners)


def corner_rmse(mask: NDArray[bool_], T: int = 1000) -> float:
    mask = mask.ravel()
    tag_pts = LATTICE[mask]

    # generate the same random noise for every tag
    rng = np.random.default_rng(2791)
    noise = rng.normal(0.0, 0.1, size=(T, 81, 2))[:, mask, :]
    noisy_img = project_all(H_true, tag_pts) + noise

    errors = []
    for img in noisy_img:
        H_est = dlt(tag_pts, img)

        est = project_all(H_est, outer_corners)

        d2 = ((est - true) ** 2).sum(axis=1)
        rmse = np.sqrt(d2.mean())

        errors.append(rmse)

    return float(np.sqrt(np.mean(np.square(errors))))


def sweep_tags(tag_ids: list[int], T: int = 1000) -> tuple[NDArray[np.int_], NDArray[float64]]:
    counts = np.array([int(tag_corner_matrix(i).sum()) for i in tag_ids])
    rmses = np.array([corner_rmse(tag_corner_matrix(i), T) for i in tag_ids])
    return counts, rmses


def plot_sweep(counts: NDArray[np.int_], rmses: NDArray[float64], tag_ids: list[int]) -> None:
    plt.scatter(counts, rmses, s=10)
    for i, n, r in zip(tag_ids, counts, rmses, strict=True):
        plt.annotate(str(i), (n, r), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("corners on tag")
    plt.ylabel("outer-corner RMSE (px)")
    plt.show()


def sweep_summary(
    tag_ids: list[int], counts: NDArray[np.int_], rmses: NDArray[float64], base: float, top: int = 5
) -> str:
    ids = np.asarray(tag_ids)
    ratio = rmses / base
    order = np.argsort(rmses)  # best (lowest RMSE) first
    r = np.corrcoef(counts, rmses)[0, 1]

    lines = [
        f"Tags tested: {len(ids)}    4-corner baseline: {base:.4f} px",
        "",
        f"{'':<10} {'RMSE (px)':>10} {'vs 4-corner':>12} {'corners':>8}",
        "-" * 43,
    ]
    for label, i in [("best", order[0]), ("median", order[len(order) // 2]), ("worst", order[-1])]:
        lines.append(f"{label:<10} {rmses[i]:>10.4f} {ratio[i]:>12.0%} {counts[i]:>8}")

    lines += [
        "",
        f"corners per tag: {counts.min()}-{counts.max()}    corr(corners, RMSE) = {r:+.2f}",
        "",
        f"{'rank':<6} {'tag':>5} {'corners':>8} {'RMSE (px)':>10} {'vs 4-corner':>12}",
        "-" * 45,
    ]
    # top and bottom `top` tags; everything if the list is short
    shown = [*list(order[:top]), None, *list(order[-top:])] if len(order) > 2 * top else list(order)
    for k in shown:
        if k is None:
            lines.append("...")
            continue
        rank = int(np.where(order == k)[0][0]) + 1
        lines.append(f"{rank:<6} {ids[k]:>5} {counts[k]:>8} {rmses[k]:>10.4f} {ratio[k]:>12.0%}")

    return "\n".join(lines)


def all_tag_test(plot: bool = False, T: int = 1000) -> str:
    tag_ids = list(range(1, 33))
    counts, rmses = sweep_tags(tag_ids, T)
    base = corner_rmse(OUTER_CORNER_MASK, T)

    if plot:
        plot_sweep(counts, rmses, tag_ids)

    return sweep_summary(tag_ids, counts, rmses, base)


def main() -> None:
    tag_id = 7
    corner_matrix = tag_corner_matrix(tag_id)

    tests = [
        ("4 outer corners", OUTER_CORNER_MASK),
        (f"tag {tag_id} corners", corner_matrix),
    ]

    print(f"{'point set':<20} {'N':>4} {'RMSE (px)':>10} {'vs 4-corner':>12}")
    print("-" * 49)
    base = None
    for name, pts in tests:
        rmse = corner_rmse(pts)
        base = base or rmse
        print(f"{name:<20} {len(pts):>4} {rmse:>10.4f} {rmse / base:>11.0%}")

    print()
    print(all_tag_test())


if __name__ == "__main__":
    main()
