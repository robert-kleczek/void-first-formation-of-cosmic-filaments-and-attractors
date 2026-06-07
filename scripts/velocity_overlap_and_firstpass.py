from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree


BASE = Path("REAL_VELOCITY_FIELDS_20260606")
VOID_PATH = Path("REAL_SDSS_VOID_LENSING") / "real_sdss_void_catalog.csv"
H = 0.70
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.27, Tcmb0=2.725)
X_WALL = 1.5507603954646239
SIGMA_WALL = 0.5776996523125006
C_KMS = 299792.458


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
    l = gal.l.to_value(u.rad)
    b = gal.b.to_value(u.rad)
    cb = np.cos(b)
    return np.column_stack(
        [
            dist_hmpc * cb * np.cos(l),
            dist_hmpc * cb * np.sin(l),
            dist_hmpc * np.sin(b),
        ]
    )


def gal_dist_to_xyz_hmpc(l_deg: np.ndarray, b_deg: np.ndarray, dist_hmpc: np.ndarray) -> np.ndarray:
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


def shell_score(q: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * ((q - X_WALL) / SIGMA_WALL) ** 2)


def interp_velocity(points: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spacing = 400.0 / 256.0
    idx = points / spacing + 128.0
    valid = np.all((idx >= 0.0) & (idx <= 256.0), axis=1)
    out = np.full((len(points), 3), np.nan, dtype=float)
    if not np.any(valid):
        return out, valid

    iv = idx[valid]
    i0 = np.floor(iv).astype(int)
    i0 = np.clip(i0, 0, 255)
    t = iv - i0
    accum = np.zeros((iv.shape[0], 3), dtype=float)
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
                accum += velocity[:, ii, jj, kk].T * w[:, None]
    out[valid] = accum
    return out, valid


def summarize_values(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p16": float(np.percentile(x, 16)),
        "p84": float(np.percentile(x, 84)),
        "frac_positive": float(np.mean(x > 0.0)),
    }


def load_voids() -> pd.DataFrame:
    voids = pd.read_csv(VOID_PATH)
    voids["x_hmpc"], voids["y_hmpc"], voids["z_hmpc"] = radec_z_to_gal_xyz_hmpc(
        voids["ra"].to_numpy(float),
        voids["dec"].to_numpy(float),
        voids["redshift"].to_numpy(float),
    ).T
    voids["dist_hmpc"] = np.linalg.norm(voids[["x_hmpc", "y_hmpc", "z_hmpc"]].to_numpy(float), axis=1)
    return voids


def carrick_void_shell_test(voids: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    velocity = np.load(BASE / "carrick_2mpp" / "twompp_velocity.npy", mmap_mode="r")
    centers = voids[["x_hmpc", "y_hmpc", "z_hmpc"]].to_numpy(float)
    radii = voids["radius"].to_numpy(float)
    dirs = fibonacci_sphere(96)

    center_velocity, center_valid = interp_velocity(centers, velocity)
    rows = []
    for idx, row in voids.iterrows():
        if not center_valid[idx]:
            continue
        c = centers[idx]
        r_shell = X_WALL * radii[idx]
        shell_points = c[None, :] + r_shell * dirs
        v_shell, valid = interp_velocity(shell_points, velocity)
        if valid.sum() < 24:
            continue
        d = dirs[valid]
        raw_out = np.einsum("ij,ij->i", v_shell[valid], d)
        rel_out = np.einsum("ij,ij->i", v_shell[valid] - center_velocity[idx][None, :], d)
        rows.append(
            {
                "void_id": row["id"],
                "redshift": row["redshift"],
                "radius_hmpc_assumed": row["radius"],
                "center_dist_hmpc": row["dist_hmpc"],
                "valid_shell_samples": int(valid.sum()),
                "raw_shell_outflow_mean_kms": float(np.mean(raw_out)),
                "relative_shell_outflow_mean_kms": float(np.mean(rel_out)),
                "relative_shell_outflow_median_kms": float(np.median(rel_out)),
                "relative_shell_frac_positive": float(np.mean(rel_out > 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "carrick_2mpp" / "carrick_void_shell_outflow_firstpass.csv", index=False)
    summary = {
        "voids_total": int(len(voids)),
        "void_centers_inside_carrick_cube": int(center_valid.sum()),
        "voids_with_usable_shell_sampling": int(len(out)),
        "relative_shell_outflow_mean_per_void_kms": summarize_values(out["relative_shell_outflow_mean_kms"].to_numpy(float) if len(out) else []),
        "relative_shell_outflow_median_per_void_kms": summarize_values(out["relative_shell_outflow_median_kms"].to_numpy(float) if len(out) else []),
        "relative_shell_frac_positive_per_void": summarize_values((out["relative_shell_frac_positive"].to_numpy(float) - 0.5) if len(out) else []),
        "notes": [
            "Carrick shell test subtracts the interpolated velocity at the void centre from shell velocities to suppress coherent bulk motion.",
            "Positive relative_shell_outflow_mean_kms means reconstructed flow is moving away from the void centre on the DRND shell.",
        ],
    }
    return out, summary


def nearest_void_shell_projection(objects: pd.DataFrame, xyz: np.ndarray, vlos: np.ndarray, voids: pd.DataFrame, label: str) -> tuple[pd.DataFrame, dict]:
    void_xyz = voids[["x_hmpc", "y_hmpc", "z_hmpc"]].to_numpy(float)
    void_r = voids["radius"].to_numpy(float)
    tree = cKDTree(void_xyz)
    dist, nearest = tree.query(xyz, k=1)
    q = dist / void_r[nearest]
    score = shell_score(q)
    radial = xyz - void_xyz[nearest]
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_unit = radial / np.maximum(radial_norm[:, None], 1e-12)
    los_unit = xyz / np.maximum(np.linalg.norm(xyz, axis=1)[:, None], 1e-12)
    los_projection = np.einsum("ij,ij->i", radial_unit, los_unit)
    outward_proxy = vlos * los_projection

    out = pd.DataFrame(
        {
            "nearest_void_row": nearest,
            "nearest_void_id": voids["id"].to_numpy()[nearest],
            "q_D_over_Rv": q,
            "ejection_score": score,
            "los_radial_projection": los_projection,
            "vlos_kms": vlos,
            "outward_velocity_proxy_kms": outward_proxy,
        }
    )
    keep = np.isfinite(outward_proxy) & np.isfinite(score)
    out = out.loc[keep].copy()
    out.to_csv(BASE / f"{label}_nearest_void_shell_velocity_projection.csv", index=False)

    cuts = {
        "all": np.ones(len(out), dtype=bool),
        "score_gt_0p5": out["ejection_score"].to_numpy(float) > 0.5,
        "score_gt_0p8": out["ejection_score"].to_numpy(float) > 0.8,
        "score_gt_0p8_abs_losproj_gt_0p2": (out["ejection_score"].to_numpy(float) > 0.8)
        & (np.abs(out["los_radial_projection"].to_numpy(float)) > 0.2),
    }
    stats = {}
    for name, mask in cuts.items():
        sub = out.loc[mask]
        if len(sub) == 0:
            stats[name] = {"n": 0}
            continue
        vals = sub["outward_velocity_proxy_kms"].to_numpy(float)
        by_void = sub.groupby("nearest_void_id")["outward_velocity_proxy_kms"].mean().to_numpy(float)
        stats[name] = {
            "objects": summarize_values(vals),
            "void_block_means": summarize_values(by_void),
            "median_q": float(np.median(sub["q_D_over_Rv"])),
            "median_abs_los_projection": float(np.median(np.abs(sub["los_radial_projection"]))),
        }
    return out, {"label": label, "rows": int(len(out)), "cuts": stats}


def sdss_pv_projection(voids: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(BASE / "sdss_pv" / "SDSS_PV_public_slim.csv")
    xyz = radec_z_to_gal_xyz_hmpc(df["RA"].to_numpy(float), df["Dec"].to_numpy(float), df["zcmb"].to_numpy(float))
    vlos = df["vpec_los_approx_kms"].to_numpy(float)
    return nearest_void_shell_projection(df, xyz, vlos, voids, "sdss_pv")


def cf4_projection(voids: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(BASE / "cosmicflows4_vizier" / "cf4_groups_slim.csv")
    raw = pd.read_csv(BASE / "cosmicflows4_vizier" / "cf4_groups_peculiar_velocities.tsv", sep="\t", comment="#", dtype=str, engine="python")
    first_col = raw.columns[0]
    raw = raw.loc[np.isfinite(pd.to_numeric(raw[first_col], errors="coerce"))].copy()
    for col in raw.columns:
        conv = pd.to_numeric(raw[col], errors="coerce")
        if conv.notna().sum() > 0:
            raw[col] = conv
    need = raw[["RAJ2000", "DEJ2000", "Dist", "Vpec"]].dropna().copy()
    dist_hmpc = need["Dist"].to_numpy(float) * H
    coord = SkyCoord(ra=need["RAJ2000"].to_numpy(float) * u.deg, dec=need["DEJ2000"].to_numpy(float) * u.deg, frame="icrs")
    gal = coord.galactic
    xyz = gal_dist_to_xyz_hmpc(gal.l.to_value(u.deg), gal.b.to_value(u.deg), dist_hmpc)
    vlos = need["Vpec"].to_numpy(float)
    return nearest_void_shell_projection(need, xyz, vlos, voids, "cf4_groups")


def main() -> None:
    voids = load_voids()
    voids.to_csv(BASE / "sdss_voids_with_galactic_xyz_hmpc.csv", index=False)
    _, carrick_summary = carrick_void_shell_test(voids)
    _, sdss_summary = sdss_pv_projection(voids)
    _, cf4_summary = cf4_projection(voids)

    summary = {
        "void_catalog": {
            "rows": int(len(voids)),
            "path": str(VOID_PATH),
            "distance_hmpc_median": float(voids["dist_hmpc"].median()),
            "distance_hmpc_range": [float(voids["dist_hmpc"].min()), float(voids["dist_hmpc"].max())],
            "radius_median_hmpc_assumed": float(voids["radius"].median()),
        },
        "carrick_2mpp_void_shell": carrick_summary,
        "sdss_pv_los_shell_projection": sdss_summary,
        "cosmicflows4_group_los_shell_projection": cf4_summary,
        "interpretation_guardrails": [
            "Carrick is a reconstructed vector field from a density model; it tests dynamic consistency of outflow geometry, not independent proof against gravitational collapse.",
            "SDSS PV line-of-sight velocities are approximate here because the rigorous observable is log-distance ratio with catalogue likelihood.",
            "Cosmicflows-4 has direct Vpec, but sky/selection mismatch with the SDSS void catalogue can bias nearest-void projections.",
        ],
    }
    (BASE / "velocity_overlap_firstpass_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Velocity overlap and first-pass dynamics",
        "",
        "## Void coverage",
        f"- SDSS voids: {summary['void_catalog']['rows']}",
        f"- median void distance: {summary['void_catalog']['distance_hmpc_median']:.1f} Mpc/h",
        f"- Carrick usable shell voids: {carrick_summary['voids_with_usable_shell_sampling']}",
        "",
        "## Carrick / 2M++ shell outflow",
        f"- mean relative shell outflow: {carrick_summary['relative_shell_outflow_mean_per_void_kms']['mean']:.1f} km/s",
        f"- median of per-void mean shell outflow: {carrick_summary['relative_shell_outflow_mean_per_void_kms']['median']:.1f} km/s",
        f"- fraction of voids with positive mean shell outflow: {carrick_summary['relative_shell_outflow_mean_per_void_kms']['frac_positive']:.3f}",
        "",
        "## SDSS PV line-of-sight projection",
        f"- shell score > 0.8 objects: {sdss_summary['cuts']['score_gt_0p8']['objects']['n']}",
        f"- score > 0.8 mean outward proxy: {sdss_summary['cuts']['score_gt_0p8']['objects']['mean']:.1f} km/s",
        f"- score > 0.8 positive fraction: {sdss_summary['cuts']['score_gt_0p8']['objects']['frac_positive']:.3f}",
        "",
        "## Cosmicflows-4 group line-of-sight projection",
        f"- shell score > 0.8 groups: {cf4_summary['cuts']['score_gt_0p8']['objects']['n']}",
        f"- score > 0.8 mean outward proxy: {cf4_summary['cuts']['score_gt_0p8']['objects']['mean']:.1f} km/s",
        f"- score > 0.8 positive fraction: {cf4_summary['cuts']['score_gt_0p8']['objects']['frac_positive']:.3f}",
        "",
    ]
    (BASE / "velocity_overlap_firstpass_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
