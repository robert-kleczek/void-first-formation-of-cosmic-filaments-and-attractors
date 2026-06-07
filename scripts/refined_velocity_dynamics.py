from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp


BASE = Path("REAL_VELOCITY_FIELDS_20260606")
OUT = BASE / "refined_dynamics"
OUT.mkdir(parents=True, exist_ok=True)

VOID_PATH = Path("REAL_SDSS_VOID_LENSING") / "real_sdss_void_catalog.csv"
H = 0.70
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.27, Tcmb0=2.725)
C_KMS = 299792.458
X_WALL = 1.5507603954646239
SIGMA_WALL = 0.5776996523125006


def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def radec_z_to_gal_xyz_hmpc(ra_deg: np.ndarray, dec_deg: np.ndarray, z: np.ndarray) -> np.ndarray:
    dist_hmpc = COSMO.comoving_distance(z).to_value(u.Mpc) * H
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    gal = coord.galactic
    return gal_spherical_to_xyz(gal.l.to_value(u.deg), gal.b.to_value(u.deg), dist_hmpc)


def gal_spherical_to_xyz(l_deg: np.ndarray, b_deg: np.ndarray, dist_hmpc: np.ndarray) -> np.ndarray:
    l = np.deg2rad(l_deg)
    b = np.deg2rad(b_deg)
    cb = np.cos(b)
    return np.column_stack(
        [
            dist_hmpc * cb * np.cos(l),
            dist_hmpc * cb * np.sin(l),
            dist_hmpc * np.sin(b),
        ]
    )


def load_voids() -> pd.DataFrame:
    voids = pd.read_csv(VOID_PATH)
    xyz = radec_z_to_gal_xyz_hmpc(
        voids["ra"].to_numpy(float),
        voids["dec"].to_numpy(float),
        voids["redshift"].to_numpy(float),
    )
    voids["x_hmpc"], voids["y_hmpc"], voids["z_hmpc"] = xyz.T
    voids["dist_hmpc"] = np.linalg.norm(xyz, axis=1)
    return voids


def interp_grid(points: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spacing = 400.0 / 256.0
    idx = points / spacing + 128.0
    valid = np.all((idx >= 0.0) & (idx <= 256.0), axis=1)

    vector = grid.ndim == 4
    shape = (len(points), 3) if vector else (len(points),)
    out = np.full(shape, np.nan, dtype=float)
    if not np.any(valid):
        return out, valid

    iv = idx[valid]
    i0 = np.floor(iv).astype(int)
    i0 = np.clip(i0, 0, 255)
    t = iv - i0
    accum = np.zeros((iv.shape[0], 3), dtype=float) if vector else np.zeros(iv.shape[0], dtype=float)
    for dx in (0, 1):
        wx = (1.0 - t[:, 0]) if dx == 0 else t[:, 0]
        for dy in (0, 1):
            wy = (1.0 - t[:, 1]) if dy == 0 else t[:, 1]
            for dz in (0, 1):
                wz = (1.0 - t[:, 2]) if dz == 0 else t[:, 2]
                w = wx * wy * wz
                ii = i0[:, 0] + dx
                jj = i0[:, 1] + dy
                kk = i0[:, 2] + dz
                if vector:
                    accum += grid[:, ii, jj, kk].T * w[:, None]
                else:
                    accum += grid[ii, jj, kk] * w
    out[valid] = accum
    return out, valid


def l012_basis(n: np.ndarray) -> np.ndarray:
    x, y, z = n[:, 0], n[:, 1], n[:, 2]
    return np.column_stack(
        [
            np.ones(len(n)),
            x,
            y,
            z,
            x * x - z * z,
            y * y - z * z,
            x * y,
            x * z,
            y * z,
        ]
    )


def fit_l012(n: np.ndarray, values: np.ndarray) -> dict:
    b = l012_basis(n)
    coeff, *_ = np.linalg.lstsq(b, values, rcond=None)
    mono = np.full_like(values, coeff[0])
    dip = b[:, 1:4] @ coeff[1:4]
    quad = b[:, 4:] @ coeff[4:]
    resid = values - (mono + dip + quad)
    return {
        "monopole": float(coeff[0]),
        "dipole_rms": float(np.sqrt(np.mean(dip * dip))),
        "quadrupole_rms": float(np.sqrt(np.mean(quad * quad))),
        "residual_rms": float(np.sqrt(np.mean(resid * resid))),
        "raw_mean": float(np.mean(values)),
    }


def summarize(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    t = ttest_1samp(x, 0.0, alternative="greater") if x.size > 1 else None
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p16": float(np.percentile(x, 16)),
        "p84": float(np.percentile(x, 84)),
        "frac_positive": float(np.mean(x > 0.0)),
        "t_p_greater_zero": float(t.pvalue) if t is not None else None,
    }


def shell_score(q: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((q - X_WALL) / SIGMA_WALL) ** 2)


def carrick_refined(voids: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    velocity = np.load(BASE / "carrick_2mpp" / "twompp_velocity.npy", mmap_mode="r")
    density = np.load(BASE / "carrick_2mpp" / "twompp_density.npy", mmap_mode="r")

    centers = voids[["x_hmpc", "y_hmpc", "z_hmpc"]].to_numpy(float)
    radii = voids["radius"].to_numpy(float)
    dirs = fibonacci_sphere(256)
    x_values = [0.8, 1.0, 1.25, X_WALL, 2.0, 2.5]

    center_v, center_valid = interp_grid(centers, velocity)
    center_delta, density_valid = interp_grid(centers, density)
    rows = []

    for idx, row in voids.iterrows():
        if not (center_valid[idx] and density_valid[idx]):
            continue
        c = centers[idx]
        r_void = radii[idx]
        v_c = center_v[idx]
        env_delta = {}
        for xmul in x_values:
            shell_points = c[None, :] + (xmul * r_void) * dirs
            v_shell, v_valid = interp_grid(shell_points, velocity)
            d_shell, d_valid = interp_grid(shell_points, density)
            valid = v_valid & d_valid
            if valid.sum() < 80:
                continue
            n = dirs[valid]
            rel_rad = np.einsum("ij,ij->i", v_shell[valid] - v_c[None, :], n)
            fit = fit_l012(n, rel_rad)
            delta_vals = d_shell[valid]
            env_delta[f"{xmul:.3f}"] = float(np.mean(delta_vals))
            rows.append(
                {
                    "void_id": row["id"],
                    "void_index": int(idx),
                    "redshift": row["redshift"],
                    "radius_hmpc_assumed": r_void,
                    "center_dist_hmpc": row["dist_hmpc"],
                    "x_over_Rv": xmul,
                    "valid_shell_samples": int(valid.sum()),
                    "center_delta_gstar": float(center_delta[idx]),
                    "shell_delta_mean": float(np.mean(delta_vals)),
                    "shell_delta_median": float(np.median(delta_vals)),
                    "shell_delta_p84": float(np.percentile(delta_vals, 84)),
                    "monopole_outflow_kms": fit["monopole"],
                    "raw_mean_outflow_kms": fit["raw_mean"],
                    "dipole_rms_kms": fit["dipole_rms"],
                    "quadrupole_rms_kms": fit["quadrupole_rms"],
                    "residual_rms_kms": fit["residual_rms"],
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, {}

    pivot_env = df.pivot_table(index="void_id", columns="x_over_Rv", values="shell_delta_mean", aggfunc="mean")
    env_x = 2.5 if 2.5 in pivot_env.columns else max(pivot_env.columns)
    void_env = pivot_env[[env_x]].rename(columns={env_x: "outer_delta_2p5R"})
    df = df.merge(void_env, left_on="void_id", right_index=True, how="left")
    df["void_class"] = np.where(df["outer_delta_2p5R"] < 0.0, "void_in_void", "void_in_cloud_or_compensated")
    q_quad = df.loc[np.isclose(df["x_over_Rv"], X_WALL), "quadrupole_rms_kms"].quantile(0.5)
    df["low_quadrupole_shear"] = df["quadrupole_rms_kms"] <= q_quad
    df.to_csv(OUT / "carrick_l012_shell_profile.csv", index=False)

    summary: dict[str, dict] = {}
    for xmul in x_values:
        sub = df[np.isclose(df["x_over_Rv"], xmul)]
        if len(sub) == 0:
            continue
        x_key = f"x_{xmul:.3f}".replace(".", "p")
        summary[x_key] = {
            "all": summarize(sub["monopole_outflow_kms"].to_numpy(float)),
            "void_in_void_outer_delta_lt_0": summarize(
                sub.loc[sub["void_class"] == "void_in_void", "monopole_outflow_kms"].to_numpy(float)
            ),
            "void_in_cloud_or_compensated_outer_delta_ge_0": summarize(
                sub.loc[sub["void_class"] != "void_in_void", "monopole_outflow_kms"].to_numpy(float)
            ),
            "low_quadrupole_shear": summarize(
                sub.loc[sub["low_quadrupole_shear"], "monopole_outflow_kms"].to_numpy(float)
            ),
            "density": {
                "center_delta": summarize(sub["center_delta_gstar"].to_numpy(float)),
                "shell_delta": summarize(sub["shell_delta_mean"].to_numpy(float)),
                "outer_delta_2p5R": summarize(sub["outer_delta_2p5R"].to_numpy(float)),
            },
            "mode_rms": {
                "dipole": summarize(sub["dipole_rms_kms"].to_numpy(float)),
                "quadrupole": summarize(sub["quadrupole_rms_kms"].to_numpy(float)),
                "residual": summarize(sub["residual_rms_kms"].to_numpy(float)),
            },
        }

    profile = (
        df.groupby(["x_over_Rv", "void_class"])["monopole_outflow_kms"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    profile.to_csv(OUT / "carrick_monopole_profile_by_void_class.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for klass, color in [("void_in_void", "tab:blue"), ("void_in_cloud_or_compensated", "tab:red")]:
        sub = df[df["void_class"] == klass]
        grp = sub.groupby("x_over_Rv")["monopole_outflow_kms"]
        xs = np.array([k for k, _ in grp], dtype=float)
        means = np.array([g.mean() for _, g in grp], dtype=float)
        err = np.array([g.std(ddof=1) / np.sqrt(len(g)) for _, g in grp], dtype=float)
        ax.errorbar(xs, means, yerr=err, marker="o", lw=1.5, capsize=3, label=klass, color=color)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(X_WALL, color="gray", lw=1, ls="--", label="x_wall")
    ax.set_xlabel("D / Rv")
    ax.set_ylabel("l=0 relative radial velocity [km/s]")
    ax.set_title("Carrick/2M++ l=0 shell outflow after density selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "carrick_l0_profile_by_void_class.png", dpi=180)
    plt.close(fig)

    return df, {
        "rows": int(len(df)),
        "voids_usable": int(df["void_id"].nunique()),
        "x_wall": X_WALL,
        "outer_density_classification": {
            "void_in_void_outer_delta_lt_0": int(
                df.loc[np.isclose(df["x_over_Rv"], X_WALL)]
                .query("void_class == 'void_in_void'")["void_id"]
                .nunique()
            ),
            "void_in_cloud_or_compensated_outer_delta_ge_0": int(
                df.loc[np.isclose(df["x_over_Rv"], X_WALL)]
                .query("void_class != 'void_in_void'")["void_id"]
                .nunique()
            ),
        },
        "by_shell": summary,
        "notes": [
            "l=0 is fitted jointly with l=1 and l=2 real angular basis terms; this isolates the monopole when shell sampling is incomplete near cube boundaries.",
            "void_in_void is operationally defined as mean Carrick delta_g* at 2.5 Rv below zero.",
        ],
    }


def matched_amplitude(vlos: np.ndarray, cosang: np.ndarray) -> float:
    denom = np.sum(cosang * cosang)
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(vlos * cosang) / denom)


def pv_refined(label: str, xyz: np.ndarray, vlos: np.ndarray, voids: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    void_xyz = voids[["x_hmpc", "y_hmpc", "z_hmpc"]].to_numpy(float)
    void_r = voids["radius"].to_numpy(float)
    tree = cKDTree(void_xyz)
    dist, nearest = tree.query(xyz, k=1)
    q = dist / void_r[nearest]
    score = shell_score(q)
    radial = xyz - void_xyz[nearest]
    radial_unit = radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
    los_unit = xyz / np.maximum(np.linalg.norm(xyz, axis=1)[:, None], 1e-12)
    cosang = np.einsum("ij,ij->i", radial_unit, los_unit)
    valid = np.isfinite(vlos) & np.isfinite(cosang) & np.isfinite(score)
    df = pd.DataFrame(
        {
            "nearest_void_id": voids["id"].to_numpy()[nearest],
            "q_D_over_Rv": q,
            "ejection_score": score,
            "los_radial_cos": cosang,
            "vlos_kms": vlos,
            "matched_numerator_kms": vlos * cosang,
            "projected_vout_kms": np.where(np.abs(cosang) > 0, vlos / cosang, np.nan),
        }
    ).loc[valid].copy()
    df.to_csv(OUT / f"{label}_pv_geometric_filter.csv", index=False)

    cuts = {
        "score_gt_0p8": df["ejection_score"].to_numpy(float) > 0.8,
        "score_gt_0p8_abs_cos_gt_0p5": (df["ejection_score"].to_numpy(float) > 0.8)
        & (np.abs(df["los_radial_cos"].to_numpy(float)) > 0.5),
        "score_gt_0p8_abs_cos_gt_0p8": (df["ejection_score"].to_numpy(float) > 0.8)
        & (np.abs(df["los_radial_cos"].to_numpy(float)) > 0.8),
        "score_gt_0p5_abs_cos_gt_0p8": (df["ejection_score"].to_numpy(float) > 0.5)
        & (np.abs(df["los_radial_cos"].to_numpy(float)) > 0.8),
    }
    summary = {}
    for name, mask in cuts.items():
        sub = df.loc[mask]
        if len(sub) == 0:
            summary[name] = {"n": 0}
            continue
        grouped = []
        for vid, g in sub.groupby("nearest_void_id"):
            if len(g) < 2:
                continue
            grouped.append(
                {
                    "nearest_void_id": vid,
                    "n": len(g),
                    "A_matched_kms": matched_amplitude(
                        g["vlos_kms"].to_numpy(float), g["los_radial_cos"].to_numpy(float)
                    ),
                }
            )
        gdf = pd.DataFrame(grouped)
        gdf.to_csv(OUT / f"{label}_{name}_void_block_matched.csv", index=False)
        summary[name] = {
            "objects": summarize(sub["matched_numerator_kms"].to_numpy(float)),
            "projected_vout_objects": summarize(sub["projected_vout_kms"].to_numpy(float)),
            "matched_amplitude_all_objects_kms": matched_amplitude(
                sub["vlos_kms"].to_numpy(float), sub["los_radial_cos"].to_numpy(float)
            ),
            "void_block_A_matched_kms": summarize(
                gdf["A_matched_kms"].to_numpy(float) if len(gdf) else np.array([])
            ),
            "n_void_blocks_with_at_least_2_objects": int(len(gdf)),
            "median_abs_cos": float(np.median(np.abs(sub["los_radial_cos"]))),
            "median_q": float(np.median(sub["q_D_over_Rv"])),
        }
    return df, {"label": label, "rows": int(len(df)), "cuts": summary}


def run_pv_refined(voids: pd.DataFrame) -> dict:
    sdss = pd.read_csv(BASE / "sdss_pv" / "SDSS_PV_public_slim.csv")
    sdss_xyz = radec_z_to_gal_xyz_hmpc(sdss["RA"].to_numpy(float), sdss["Dec"].to_numpy(float), sdss["zcmb"].to_numpy(float))
    _, sdss_summary = pv_refined("sdss_pv", sdss_xyz, sdss["vpec_los_approx_kms"].to_numpy(float), voids)

    raw = pd.read_csv(
        BASE / "cosmicflows4_vizier" / "cf4_groups_peculiar_velocities.tsv",
        sep="\t",
        comment="#",
        dtype=str,
        engine="python",
    )
    first_col = raw.columns[0]
    raw = raw.loc[np.isfinite(pd.to_numeric(raw[first_col], errors="coerce"))].copy()
    for col in raw.columns:
        conv = pd.to_numeric(raw[col], errors="coerce")
        if conv.notna().sum() > 0:
            raw[col] = conv
    cf4 = raw[["RAJ2000", "DEJ2000", "Dist", "Vpec"]].dropna().copy()
    coord = SkyCoord(ra=cf4["RAJ2000"].to_numpy(float) * u.deg, dec=cf4["DEJ2000"].to_numpy(float) * u.deg, frame="icrs")
    gal = coord.galactic
    cf4_xyz = gal_spherical_to_xyz(
        gal.l.to_value(u.deg), gal.b.to_value(u.deg), cf4["Dist"].to_numpy(float) * H
    )
    _, cf4_summary = pv_refined("cf4_groups", cf4_xyz, cf4["Vpec"].to_numpy(float), voids)
    return {"sdss_pv": sdss_summary, "cosmicflows4_groups": cf4_summary}


def write_markdown(summary: dict) -> None:
    xkey = f"x_{X_WALL:.3f}".replace(".", "p")
    carrick = summary["carrick_refined"]["by_shell"][xkey]
    sdss = summary["pv_refined"]["sdss_pv"]["cuts"]["score_gt_0p8_abs_cos_gt_0p8"]
    cf4 = summary["pv_refined"]["cosmicflows4_groups"]["cuts"]["score_gt_0p8_abs_cos_gt_0p8"]
    lines = [
        "# Refined velocity dynamics",
        "",
        "## Carrick / 2M++ l=0 shell at x_wall",
        f"- usable voids: {summary['carrick_refined']['voids_usable']}",
        f"- void-in-void count: {summary['carrick_refined']['outer_density_classification']['void_in_void_outer_delta_lt_0']}",
        f"- cloud/compensated count: {summary['carrick_refined']['outer_density_classification']['void_in_cloud_or_compensated_outer_delta_ge_0']}",
        f"- all l=0 mean: {carrick['all']['mean']:.2f} km/s; median: {carrick['all']['median']:.2f} km/s; positive fraction: {carrick['all']['frac_positive']:.3f}",
        f"- void-in-void l=0 mean: {carrick['void_in_void_outer_delta_lt_0']['mean']:.2f} km/s; median: {carrick['void_in_void_outer_delta_lt_0']['median']:.2f} km/s; positive fraction: {carrick['void_in_void_outer_delta_lt_0']['frac_positive']:.3f}",
        f"- cloud/compensated l=0 mean: {carrick['void_in_cloud_or_compensated_outer_delta_ge_0']['mean']:.2f} km/s; median: {carrick['void_in_cloud_or_compensated_outer_delta_ge_0']['median']:.2f} km/s; positive fraction: {carrick['void_in_cloud_or_compensated_outer_delta_ge_0']['frac_positive']:.3f}",
        "",
        "## PV LOS with |cos| > 0.8 and shell score > 0.8",
        f"- SDSS PV objects: {sdss['objects']['n']}; matched A_all: {sdss['matched_amplitude_all_objects_kms']:.2f} km/s; projected median: {sdss['projected_vout_objects']['median']:.2f} km/s",
        f"- SDSS void-block A median: {sdss['void_block_A_matched_kms']['median']:.2f} km/s; positive fraction: {sdss['void_block_A_matched_kms']['frac_positive']:.3f}",
        f"- CF4 groups: {cf4['objects']['n']}; matched A_all: {cf4['matched_amplitude_all_objects_kms']:.2f} km/s; projected median: {cf4['projected_vout_objects']['median']:.2f} km/s",
        f"- CF4 void-block A median: {cf4['void_block_A_matched_kms']['median']:.2f} km/s; positive fraction: {cf4['void_block_A_matched_kms']['frac_positive']:.3f}",
        "",
        "## Interpretation",
        "- Carrick is the cleanest kinematic test here because it is a vector field; SDSS PV and CF4 remain noisy LOS tests.",
        "- The density split is operational, not a final physical taxonomy: void-in-void is delta_g* at 2.5 Rv below zero.",
    ]
    (OUT / "refined_velocity_dynamics_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    voids = load_voids()
    _, carrick_summary = carrick_refined(voids)
    pv_summary = run_pv_refined(voids)
    summary = {
        "carrick_refined": carrick_summary,
        "pv_refined": pv_summary,
        "outputs": {
            "carrick_profile_csv": str(OUT / "carrick_l012_shell_profile.csv"),
            "carrick_profile_plot": str(OUT / "carrick_l0_profile_by_void_class.png"),
            "summary_md": str(OUT / "refined_velocity_dynamics_summary.md"),
        },
    }
    (OUT / "refined_velocity_dynamics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
