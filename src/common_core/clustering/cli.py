"""``bgc-cluster`` — HPC entry point.

Subcommands:
  run               full clustering pipeline (primary NRBs + partials)
  project-partials  partial-only projection (used rarely; primarily for
                    re-projecting partials against an existing primary run)

Defaults match the portal's in-process pipeline so HPC and dev runs are
comparable. Numerics are identical on the CPU path; the GPU path has
documented drift (UMAP is stochastic; cuml ≠ umap-learn bit-for-bit).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

from common_core.clustering.io import read_inputs_tarball, write_outputs_tarball
from common_core.clustering.knn import build_knn_graph
from common_core.clustering.partial import project_partials
from common_core.clustering.schema import ClusteringOutputs
from common_core.clustering.scoring import (
    annotate_gcf_nodes,
    build_ltree_paths,
    compute_domain_novelty_array,
    compute_novelty_against_validated,
)

log = logging.getLogger("bgc-cluster")


DEFAULT_RESOLUTIONS: tuple[float, ...] = (0.03, 0.08, 0.15, 0.25)
DEFAULT_WEIGHTS: tuple[float, float] = (0.5, 0.5)
KNN_K_FLOOR = 5


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "project-partials":
        return _cmd_project_partials(args)
    parser.print_help()
    return 2


# ── Subcommand: run ─────────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> int:
    device = args.device
    log.info("bgc-cluster run --device %s --input %s --output %s",
             device, args.input, args.output)

    diag = {"device": device, "cuda_version": "", "gpu_model": "",
            "library_versions": {}}
    if device == "gpu":
        # Fail-fast import fence — no silent CPU fallback.
        from common_core.clustering.gpu import ensure_gpu_available

        diag.update(ensure_gpu_available())
        log.info("GPU detected: %s (cuda=%s, libs=%s)",
                 diag["gpu_model"], diag["cuda_version"], diag["library_versions"])

    inputs = read_inputs_tarball(Path(args.input))
    log.info(
        "loaded inputs: M_domains=%s M_pairs=%s nrb_ids=%d partials=%d validated=%d",
        inputs.M_domains.shape, inputs.M_pairs.shape,
        len(inputs.nrb_ids), len(inputs.partials_nrb_ids),
        len(inputs.validated_nrb_ids),
    )

    # Pipeline parameter resolution. CLI overrides params.json when supplied.
    weights = tuple(args.weight or inputs.params.score_weights or DEFAULT_WEIGHTS)
    resolutions = tuple(args.resolutions or inputs.params.leiden_resolutions or DEFAULT_RESOLUTIONS)
    seed = args.seed if args.seed is not None else inputs.params.seed
    knn_k = args.knn_k if args.knn_k is not None else inputs.params.knn_k

    n_primary = inputs.M_domains.shape[0]
    if n_primary == 0:
        log.error("no primary NRBs in input tarball — nothing to cluster")
        return 1

    effective_k = int(knn_k) if knn_k is not None and knn_k != "auto" else _auto_knn_k(n_primary)
    log.info("kNN k=%d (n_primary=%d, supplied=%s)", effective_k, n_primary, knn_k)

    # 1. composite-Dice similarity
    sim = _compute_sim(
        inputs.M_domains, inputs.M_pairs, weights=weights,
        device=device, matmul_workers=args.matmul_workers, prune_below=0.05,
    )

    # 2. KNN graph (host)
    graph = build_knn_graph(sim, k=effective_k)

    # 3. hierarchical Leiden
    if device == "gpu":
        from common_core.clustering.gpu.leiden import run_hierarchical_leiden_gpu as _hl
    else:
        from common_core.clustering.leiden import run_hierarchical_leiden as _hl
    levels = _hl(graph, resolutions=resolutions, seed=seed)

    # 4. ltree paths + GCF tree
    leaf_paths_map, gcf_nodes = build_ltree_paths(levels, inputs.nrb_ids)
    leaf_paths = [leaf_paths_map[int(x)] for x in inputs.nrb_ids.tolist()]
    annotate_gcf_nodes(gcf_nodes, sim, inputs.nrb_ids)

    # 5. 2D layout
    if device == "gpu":
        from common_core.clustering.gpu.layout import compute_2d_layout_gpu as _layout
    else:
        from common_core.clustering.layout import compute_2d_layout as _layout
    coords = _layout(graph, sim, seed=seed)

    # 6. novelty + domain_novelty
    # Novelty is computed against a freshly-built, unpruned, diagonal-intact
    # composite-Dice block (NOT the pruned/zeroed `sim` used for KNN/Leiden):
    # a validated NRB's self-match then yields novelty 0 by construction.
    validated_cols = _validated_cols(inputs.nrb_ids, inputs.validated_nrb_ids)
    novelty = compute_novelty_against_validated(
        inputs.M_domains, inputs.M_pairs, validated_cols, weights=weights,
    )
    domain_novelty = compute_domain_novelty_array(inputs.M_domains, leaf_paths)

    # 7. partial-NRB projection (bundled here so the portal never has to run it)
    primary_coords = coords
    partial_assignments, n_skipped = project_partials(
        M_dom_pri=inputs.M_domains,
        M_pair_pri=inputs.M_pairs,
        pri_nrb_ids=inputs.nrb_ids,
        pri_coords=primary_coords,
        pri_leaf_paths=leaf_paths,
        pri_validated_rows=validated_cols,
        validated_nrb_ids=inputs.validated_nrb_ids,
        M_dom_q=inputs.partials_M_domains,
        M_pair_q=inputs.partials_M_pairs,
        partial_nrb_ids=inputs.partials_nrb_ids,
        weights=weights,
        knn_k=effective_k,
    )
    log.info(
        "partials: projected=%d skipped=%d",
        len(partial_assignments), n_skipped,
    )

    # 8. sha256 — stable, parameter-keyed
    sha = _compute_run_sha(
        sources=tuple(inputs.params.domain_sources),
        weights=weights,
        knn_k=effective_k,
        leiden_resolutions=resolutions,
        seed=seed,
        nrb_etag=f"{len(inputs.nrb_ids)}:{int(inputs.nrb_ids[-1]) if len(inputs.nrb_ids) else 0}",
        domain_etag=f"{inputs.M_domains.nnz}:{inputs.M_domains.shape[1]}",
    )

    n_root = sum(1 for n in gcf_nodes if n["level"] == 0)
    n_leaf = sum(1 for n in gcf_nodes if n["level"] == len(resolutions) - 1)
    libs = diag["library_versions"] or {}
    libs.update(
        {
            "igraph": _safe_version("igraph"),
            "leidenalg": _safe_version("leidenalg"),
            "umap-learn": _safe_version("umap-learn"),
            "scipy": _safe_version("scipy"),
            "numpy": _safe_version("numpy"),
        }
    )

    outputs = ClusteringOutputs(
        leaf_paths=leaf_paths,
        levels=levels,
        coords=coords,
        novelty_score=novelty,
        domain_novelty=domain_novelty,
        gcf_nodes=gcf_nodes,
        partial_assignments=partial_assignments,
        sha256=sha,
        n_root_communities=n_root,
        n_leaf_communities=n_leaf,
        library_versions=libs,
        device=device,
        cuda_version=diag["cuda_version"],
        gpu_model=diag["gpu_model"],
    )
    write_outputs_tarball(Path(args.output), inputs, outputs)
    log.info("Done. sha256=%s n_root=%d n_leaf=%d", sha[:12], n_root, n_leaf)
    return 0


# ── Subcommand: project-partials ────────────────────────────────────────────


def _cmd_project_partials(args: argparse.Namespace) -> int:
    """Project a fresh batch of partials against a previous primary run.

    Reads:
      --input         input tarball used by the primary run (for M_*_pri + nrb_ids)
      --partials      tarball with partial signature matrices + nrb_ids and
                      the primary run's leaf_paths and coords (parquet)
      --output        tarball with partial_assignments only

    This is a niche path; the standard ``run`` already projects partials.
    """
    import json
    import numpy as np
    import scipy.sparse as sp
    import tarfile
    import tempfile

    log.info("bgc-cluster project-partials --input %s --partials %s --output %s",
             args.input, args.partials, args.output)
    inputs = read_inputs_tarball(Path(args.input))

    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        with tarfile.open(args.partials, "r:gz") as tf:
            tf.extractall(scratch)
        meta = json.loads((scratch / "primary_meta.json").read_text())
        partials_dom = sp.load_npz(scratch / "partials_M_domains.npz")
        partials_pair = sp.load_npz(scratch / "partials_M_pairs.npz")
        partials_ids = np.load(scratch / "partials_nrb_ids.npy")

    weights = tuple(meta.get("score_weights") or DEFAULT_WEIGHTS)
    knn_k = int(meta.get("knn_k") or _auto_knn_k(inputs.M_domains.shape[0]))
    leaf_paths = list(meta.get("leaf_paths") or [])
    coords = np.asarray(meta.get("coords") or [[0.0, 0.0]] * inputs.M_domains.shape[0])
    validated_cols = _validated_cols(inputs.nrb_ids, inputs.validated_nrb_ids)

    assignments, _ = project_partials(
        M_dom_pri=inputs.M_domains,
        M_pair_pri=inputs.M_pairs,
        pri_nrb_ids=inputs.nrb_ids,
        pri_coords=coords,
        pri_leaf_paths=leaf_paths,
        pri_validated_rows=validated_cols,
        validated_nrb_ids=inputs.validated_nrb_ids,
        M_dom_q=partials_dom,
        M_pair_q=partials_pair,
        partial_nrb_ids=partials_ids,
        weights=weights,
        knn_k=knn_k,
    )

    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["nrb_id", "leaf_path", "umap_x", "umap_y", "novelty_score", "domain_novelty"]
    cols = {k: [row.get(k) for row in assignments] for k in keys}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        pq.write_table(pa.table(cols), scratch / "partial_assignments.parquet")
        with tarfile.open(out_path, "w:gz") as tf:
            tf.add(scratch / "partial_assignments.parquet", arcname="partial_assignments.parquet")
    log.info("Wrote %d partial assignments to %s", len(assignments), out_path)
    return 0


# ── Helpers ─────────────────────────────────────────────────────────────────


def _compute_sim(M_dom, M_pair, *, weights, device: str, matmul_workers: int, prune_below: float):
    if device == "gpu":
        from common_core.clustering.gpu.similarity import compute_composite_similarity_gpu

        return compute_composite_similarity_gpu(
            M_dom, M_pair, weights=weights, prune_below=prune_below,
        )
    from common_core.clustering.similarity import compute_composite_similarity

    return compute_composite_similarity(
        M_dom, M_pair, weights=weights, prune_below=prune_below,
        matmul_workers=matmul_workers,
    )


def _auto_knn_k(n: int) -> int:
    if n <= 1:
        return KNN_K_FLOOR
    return max(KNN_K_FLOOR, math.ceil(math.log(n)))


def _validated_cols(nrb_ids, validated_nrb_ids) -> list[int]:
    nrb_ids_list = [int(x) for x in nrb_ids.tolist()]
    id_to_row = {nid: i for i, nid in enumerate(nrb_ids_list)}
    validated_set = {int(x) for x in validated_nrb_ids.tolist()}
    return sorted(id_to_row[nid] for nid in validated_set if nid in id_to_row)


def _safe_version(pkg: str) -> str:
    try:
        return _pkg_version(pkg)
    except PackageNotFoundError:
        return ""


def _compute_run_sha(
    *,
    sources: tuple[str, ...],
    weights: tuple[float, float],
    knn_k: int,
    leiden_resolutions: tuple[float, ...],
    seed: int,
    nrb_etag: str,
    domain_etag: str,
) -> str:
    payload = "|".join(
        [
            f"sources={','.join(sorted(sources))}",
            f"weights={weights[0]:.6f},{weights[1]:.6f}",
            f"k={knn_k}",
            "res=" + ",".join(f"{r:.6f}" for r in leiden_resolutions),
            f"seed={seed}",
            f"nrb_etag={nrb_etag}",
            f"domain_etag={domain_etag}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Argparse ────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bgc-cluster",
        description="HPC clustering for BGC discovery (CPU + optional GPU).",
    )
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser("run", help="Run full clustering pipeline")
    pr.add_argument("--input", required=True, help="path to input tarball")
    pr.add_argument("--output", required=True, help="path to write output tarball")
    pr.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="cpu (default) or gpu (RAPIDS — requires the hpc-gpu extras and a CUDA node)",
    )
    pr.add_argument("--knn-k", default=None, help="KNN k; default 'auto' = max(5, ceil(ln(n)))")
    pr.add_argument(
        "--resolutions", nargs="+", type=float, default=None,
        help="Leiden CPM resolutions, coarsest first (default 0.03 0.08 0.15 0.25)",
    )
    pr.add_argument(
        "--weight", nargs=2, type=float, default=None,
        metavar=("W_DOMAIN", "W_ADJACENCY"),
        help="Composite Dice weights (default 0.5 0.5)",
    )
    pr.add_argument("--seed", type=int, default=None, help="Reproducibility seed")
    pr.add_argument(
        "--matmul-workers", type=int, default=1,
        help=(
            "Row-chunked sparse-matmul worker count. Default 1 (single-threaded). "
            "Only set >1 at N≥100k; loky overhead dominates below that."
        ),
    )

    pp = sub.add_parser("project-partials", help="Project partials against a previous run")
    pp.add_argument("--input", required=True)
    pp.add_argument("--partials", required=True)
    pp.add_argument("--output", required=True)
    pp.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    return p


if __name__ == "__main__":
    raise SystemExit(main())
