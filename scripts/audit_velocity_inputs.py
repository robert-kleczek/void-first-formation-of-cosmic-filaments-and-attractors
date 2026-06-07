from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


C_KMS = 299792.458
BASE = Path("REAL_VELOCITY_FIELDS_20260606")


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> dict:
    out: dict[str, dict[str, float | int]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        out[col] = {
            "n": int(x.size),
            "min": float(np.min(x)),
            "p16": float(np.percentile(x, 16)),
            "median": float(np.median(x)),
            "mean": float(np.mean(x)),
            "p84": float(np.percentile(x, 84)),
            "max": float(np.max(x)),
        }
    return out


def read_sdss_pv(path: Path) -> tuple[pd.DataFrame, dict]:
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    columns = first.lstrip("#").split()
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=columns, engine="python")

    for col in [
        "RA",
        "Dec",
        "l",
        "b",
        "zhelio",
        "zcmb",
        "SIGMA_STARS",
        "logdist",
        "logdist_err",
        "logdist_corr",
        "logdist_corr_err",
        "NgroupT17",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Low-z sanity conversion from log-distance ratio. This is not a replacement
    # for the catalogue likelihood; it is only a compact sign/magnitude proxy.
    if {"zcmb", "logdist_corr"}.issubset(df.columns):
        df["vpec_los_approx_kms"] = C_KMS * df["zcmb"] * (1.0 - 10.0 ** (-df["logdist_corr"]))
        df["vobs_cmb_kms"] = C_KMS * df["zcmb"]

    slim_cols = [
        "PGC",
        "objid",
        "specObjID",
        "RA",
        "Dec",
        "l",
        "b",
        "zhelio",
        "zcmb",
        "IDgroupT17",
        "NgroupT17",
        "logdist_corr",
        "logdist_corr_err",
        "vpec_los_approx_kms",
        "vobs_cmb_kms",
    ]
    slim_cols = [c for c in slim_cols if c in df.columns]
    df[slim_cols].to_csv(BASE / "sdss_pv" / "SDSS_PV_public_slim.csv", index=False)

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "numeric": numeric_summary(
            df,
            [
                "RA",
                "Dec",
                "l",
                "b",
                "zhelio",
                "zcmb",
                "SIGMA_STARS",
                "logdist_corr",
                "logdist_corr_err",
                "vpec_los_approx_kms",
                "vobs_cmb_kms",
                "NgroupT17",
            ],
        ),
        "notes": [
            "SDSS PV is a Fundamental Plane peculiar-velocity catalogue.",
            "The catalogue provides log-distance-ratio quantities; vpec_los_approx_kms is a low-z derived proxy for quick directional tests, not the final likelihood estimator.",
        ],
    }
    return df, summary


def read_vizier_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str, engine="python")
    first_col = df.columns[0]
    numeric_first = pd.to_numeric(df[first_col], errors="coerce")
    df = df.loc[np.isfinite(numeric_first)].copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    return df


def read_cf4() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    indiv = read_vizier_tsv(BASE / "cosmicflows4_vizier" / "cf4_table2_individual_galaxies.tsv")
    groups = read_vizier_tsv(BASE / "cosmicflows4_vizier" / "cf4_groups_peculiar_velocities.tsv")

    for name, df in [("individual", indiv), ("groups", groups)]:
        slim_candidates = [
            "recno",
            "PGC",
            "1PGC",
            "Ngal",
            "RAJ2000",
            "DEJ2000",
            "Glon",
            "Glat",
            "Vhel",
            "Vcmb",
            "Vls",
            "DMav",
            "e_DMav",
            "Dist",
            "e_Dist",
            "Vpec",
            "e_Vpec",
        ]
        slim_cols = [c for c in slim_candidates if c in df.columns]
        df[slim_cols].to_csv(BASE / "cosmicflows4_vizier" / f"cf4_{name}_slim.csv", index=False)

    interesting = [
        "RAJ2000",
        "DEJ2000",
        "Glon",
        "Glat",
        "Vhel",
        "Vcmb",
        "Vls",
        "DMav",
        "e_DMav",
        "Dist",
        "e_Dist",
        "Vpec",
        "e_Vpec",
        "Ngal",
    ]
    summary = {
        "individual_rows": int(len(indiv)),
        "group_rows": int(len(groups)),
        "individual_columns": list(indiv.columns),
        "group_columns": list(groups.columns),
        "individual_numeric": numeric_summary(indiv, interesting),
        "group_numeric": numeric_summary(groups, interesting),
        "notes": [
            "Cosmicflows-4 individual table gives galaxy distances; the groups table carries group distances and peculiar velocities.",
            "For void/filament causality tests the group table is usually safer because grouping suppresses virial/internal motions.",
        ],
    }
    return indiv, groups, summary


def read_carrick() -> dict:
    density = np.load(BASE / "carrick_2mpp" / "twompp_density.npy", mmap_mode="r")
    velocity = np.load(BASE / "carrick_2mpp" / "twompp_velocity.npy", mmap_mode="r")
    v0 = velocity[:, 128, 128, 128].astype(float)
    rng = np.random.default_rng(20260606)
    idx = rng.integers(0, velocity.shape[1], size=(25000, 3))
    vv = velocity[:, idx[:, 0], idx[:, 1], idx[:, 2]].T.astype(float)
    speed = np.linalg.norm(vv, axis=1)
    dens = density[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float)

    return {
        "density_shape": list(density.shape),
        "density_dtype": str(density.dtype),
        "velocity_shape": list(velocity.shape),
        "velocity_dtype": str(velocity.dtype),
        "grid": {
            "n_cells_per_axis": 257,
            "extent_mpc_over_h": [-200.0, 200.0],
            "spacing_mpc_over_h": 400.0 / 256.0,
            "origin_index": [128, 128, 128],
            "frame": "Galactic Cartesian, CMB-frame peculiar velocities",
        },
        "local_group_velocity_kms": {
            "vx": float(v0[0]),
            "vy": float(v0[1]),
            "vz": float(v0[2]),
            "speed": float(np.linalg.norm(v0)),
        },
        "random_sample_25000": {
            "speed_kms": {
                "min": float(speed.min()),
                "median": float(np.median(speed)),
                "mean": float(speed.mean()),
                "p84": float(np.percentile(speed, 84)),
                "max": float(speed.max()),
            },
            "density_delta_g_star": {
                "min": float(dens.min()),
                "median": float(np.median(dens)),
                "mean": float(dens.mean()),
                "p84": float(np.percentile(dens, 84)),
                "max": float(dens.max()),
            },
        },
        "notes": [
            "Carrick/2M++ is a reconstructed 3D velocity and density field, not a per-galaxy measured PV catalogue.",
            "It is directly useful for interpolating predicted peculiar velocity vectors at void centers, shell positions, and filament points inside +/-200 Mpc/h.",
        ],
    }


def read_rsd() -> dict:
    pk_path = BASE / "sdss_rsd" / "sdss_DR16_MultiTracerELGLRG_BAORSD_FS_PK.dat"
    cov_path = BASE / "sdss_rsd" / "sdss_DR16_MultiTracerELGLRG_BAORSD_FS_PK_cov.dat"
    rows = []
    for line in pk_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals, _, comment = line.partition("#")
        nums = vals.split()
        if len(nums) >= 3:
            rows.append(
                {
                    "z": float(nums[0]),
                    "value": float(nums[1]),
                    "sigma": float(nums[2]),
                    "parameter": comment.strip(),
                }
            )
    rsd = pd.DataFrame(rows)
    rsd.to_csv(BASE / "sdss_rsd" / "sdss_DR16_MultiTracerELGLRG_BAORSD_FS_PK.csv", index=False)
    cov = np.loadtxt(cov_path)
    return {
        "rows": int(len(rsd)),
        "measurements": rsd.to_dict(orient="records"),
        "covariance_shape": list(cov.shape),
        "covariance_diag": [float(x) for x in np.diag(cov)],
        "notes": [
            "This RSD product is a compressed global BAO/RSD measurement, not a position-by-position velocity catalogue.",
            "It is useful as a growth-rate consistency prior; it cannot by itself test local radial outflow from individual voids.",
        ],
    }


def main() -> None:
    sdss_df, sdss_summary = read_sdss_pv(BASE / "sdss_pv" / "SDSS_PV_public.dat")
    cf4_indiv, cf4_groups, cf4_summary = read_cf4()
    carrick_summary = read_carrick()
    rsd_summary = read_rsd()

    summary = {
        "generated_from": str(BASE),
        "sdss_pv": sdss_summary,
        "cosmicflows4_vizier": cf4_summary,
        "carrick_2mpp": carrick_summary,
        "sdss_rsd": rsd_summary,
        "next_analysis_priority": [
            "Use Carrick/2M++ vector field for direct vector-dot-radial tests around local voids within +/-200 Mpc/h.",
            "Use Cosmicflows-4 group Vpec and SDSS PV log-distance-ratio data for line-of-sight radial outflow tests after sky/volume overlap with SDSS voids.",
            "Use SDSS/eBOSS RSD only as a global growth-rate prior, not as object-level causal evidence.",
        ],
    }

    out_json = BASE / "velocity_inputs_audit_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Velocity input audit",
        "",
        "## SDSS PV",
        f"- rows: {sdss_summary['rows']}",
        f"- zcmb median: {sdss_summary['numeric']['zcmb']['median']:.5f}",
        f"- zcmb range: {sdss_summary['numeric']['zcmb']['min']:.5f} .. {sdss_summary['numeric']['zcmb']['max']:.5f}",
        f"- approx LOS PV median: {sdss_summary['numeric']['vpec_los_approx_kms']['median']:.1f} km/s",
        "",
        "## Cosmicflows-4 / VizieR",
        f"- individual galaxy rows: {cf4_summary['individual_rows']}",
        f"- group rows: {cf4_summary['group_rows']}",
    ]
    if "Vpec" in cf4_summary["group_numeric"]:
        md.extend(
            [
                f"- group Vpec median: {cf4_summary['group_numeric']['Vpec']['median']:.1f} km/s",
                f"- group Vpec p16..p84: {cf4_summary['group_numeric']['Vpec']['p16']:.1f} .. {cf4_summary['group_numeric']['Vpec']['p84']:.1f} km/s",
            ]
        )
    md.extend(
        [
            "",
            "## Carrick / 2M++",
            f"- velocity shape: {carrick_summary['velocity_shape']}",
            f"- density shape: {carrick_summary['density_shape']}",
            f"- Local Group speed: {carrick_summary['local_group_velocity_kms']['speed']:.1f} km/s",
            "",
            "## SDSS/eBOSS RSD",
            f"- measurement rows: {rsd_summary['rows']}",
            "- type: global BAO/RSD compressed measurement, not a per-object velocity catalogue",
            "",
        ]
    )
    (BASE / "velocity_inputs_audit_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({
        "sdss_pv_rows": len(sdss_df),
        "cf4_individual_rows": len(cf4_indiv),
        "cf4_group_rows": len(cf4_groups),
        "carrick_velocity_shape": carrick_summary["velocity_shape"],
        "rsd_rows": rsd_summary["rows"],
        "summary": str(out_json),
    }, indent=2))


if __name__ == "__main__":
    main()
