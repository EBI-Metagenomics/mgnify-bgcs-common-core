"""ESM-C protein embedding worker.

CLI usage:
    esmc-embed --input-file proteins.faa --output-dir out/
    esmc-embed-bgc --input-file bgc.faa --output-dir out/ --accession BGC0000001

Library usage:
    from common_core.esmc_embedder import embed_sequences, aggregate_bgc_sequences
    vecs = embed_sequences(["MKTII...", "ACDEF..."])
    # returns list of np.ndarray, shape [num_layers, hidden_dim] each

    bgc_vec = aggregate_bgc_sequences(["MKTII...", "ACDEF..."], layer=29, scale=0.5)
    # returns np.ndarray of shape (hidden_dim,)
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from Bio import SeqIO
from esm.models.esmc import ESMC
from esm.sdk import batch_executor
from esm.sdk.api import ESMProtein, LogitsConfig
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from tqdm import tqdm

from common_core.config_loader import LoaderOptions, load_settings
from common_core.logging_setup import LoggingConfig, setup_logging
from common_core.versioning import dist_version

log = logging.getLogger(__name__)

FileFormat = Literal["fasta", "genbank"]
DeviceType = Literal["cpu", "cuda"]

# LogitsConfig used for every inference call — embeddings disabled, hidden states on
_LOGITS_CFG = LogitsConfig(sequence=True, return_embeddings=False, return_hidden_states=True)


# -------------------------
# Config
# -------------------------


class ESMCConfig(BaseModel):
    model_name: str = "esmc_600m"
    device: Optional[DeviceType] = None
    np_dtype: Literal["float32", "float16"] = "float32"
    batch_token_budget: int = 6000
    max_single_sequence: int = 10000
    parquet_row_group_size: int = 10000
    retry_failed_individually: bool = True
    skip_failed_records: bool = True
    validate_sequence: bool = True


class IOConfig(BaseModel):
    input_file: str
    input_format: FileFormat = "fasta"
    output_dir: str = "out/esmc"
    output_prefix: str = "proteins"


class JobSettings(BaseSettings):
    job_name: str = "esmc-embed"
    io: IOConfig
    esmc: ESMCConfig = Field(default_factory=ESMCConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


class BGCAggregationConfig(BaseModel):
    layer: Union[int, Literal["final"]] = 29
    scale: float = 0.5
    aggregation: Literal["mean", "max"] = "mean"
    per_protein_norm: bool = False
    post_norm: bool = False
    pe_before_norm: bool = True
    write_protein_parquet: bool = True
    accession: Optional[str] = None
    output_prefix: str = "bgc"


class BGCJobSettings(BaseSettings):
    job_name: str = "esmc-embed-bgc"
    io: IOConfig
    esmc: ESMCConfig = Field(default_factory=ESMCConfig)
    bgc: BGCAggregationConfig = Field(default_factory=BGCAggregationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# -------------------------
# Helpers
# -------------------------


def _resolve_device(requested: Optional[DeviceType]) -> DeviceType:
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return requested


def _np_dtype(name: str) -> np.dtype:
    return np.float16 if name == "float16" else np.float32


def _sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# I/O
# -------------------------


def _open(path: str) -> object:
    opener = gzip.open if path.endswith(".gz") else open
    return opener(path, "rt", encoding="utf-8")


def iter_fasta_proteins(path: str) -> Iterator[Tuple[str, str]]:
    with _open(path) as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            seq = str(rec.seq)
            if seq:
                yield _sha256(seq), seq


def iter_genbank_proteins(path: str) -> Iterator[Tuple[str, str]]:
    with _open(path) as fh:
        for record in SeqIO.parse(fh, "genbank"):
            for feature in getattr(record, "features", []) or []:
                if feature.type != "CDS":
                    continue
                quals = getattr(feature, "qualifiers", {}) or {}
                translation = (quals.get("translation") or [None])[0]
                if translation:
                    seq = str(translation)
                    yield _sha256(seq), seq


def iter_proteins(path: str, fmt: FileFormat) -> Iterator[Tuple[str, str]]:
    if fmt == "fasta":
        return iter_fasta_proteins(path)
    if fmt == "genbank":
        return iter_genbank_proteins(path)
    raise ValueError(f"Unsupported format: {fmt}")


def validate_input_file(path: str, fmt: FileFormat) -> None:
    """Peek at first record only — raises on empty or malformed file."""
    bio_fmt = "fasta" if fmt == "fasta" else "genbank"
    with _open(path) as fh:
        first = next(SeqIO.parse(fh, bio_fmt), None)
    if first is None:
        raise ValueError(f"No {fmt} records found in {path}")


# -------------------------
# Validation
# -------------------------

_VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWYBXZUOJ")


def validate_protein_sequence(pid: str, seq: str) -> None:
    s = seq.strip().upper()
    if not s:
        raise ValueError(f"Empty sequence pid={pid}")
    bad = set(s) - _VALID_AA
    if bad:
        raise ValueError(f"Invalid amino acids pid={pid} bad={sorted(bad)}")


# -------------------------
# Budget batcher
# -------------------------


@dataclass
class _ProteinRecord:
    protein_id: str
    sequence: str


def _batch_by_budget(
    records: Iterator[Tuple[str, str]],
    *,
    budget: int,
    max_single_sequence: int,
) -> Iterator[List[_ProteinRecord]]:
    batch: List[_ProteinRecord] = []
    used = 0

    for pid, seq in records:
        if not seq:
            continue
        if len(seq) > max_single_sequence:
            log.warning(
                "Skipping sequence exceeding max length: pid=%s len=%d max=%d",
                pid, len(seq), max_single_sequence,
            )
            continue

        cost = len(seq)

        if not batch and cost > budget:
            yield [_ProteinRecord(pid, seq)]
            continue

        if used + cost > budget and batch:
            yield batch
            batch = [_ProteinRecord(pid, seq)]
            used = cost
        else:
            batch.append(_ProteinRecord(pid, seq))
            used += cost

    if batch:
        yield batch


# -------------------------
# Model singleton
# -------------------------

_model_cache: dict[str, ESMC] = {}


def _get_model(model_name: str, device: str) -> ESMC:
    key = f"{model_name}:{device}"
    if key not in _model_cache:
        log.info("Loading ESM-C model %s on %s", model_name, device)
        _model_cache[key] = ESMC.from_pretrained(model_name).to(device)
    return _model_cache[key]


# -------------------------
# Inference
# -------------------------


def _embed_sequence_user_func(
    client: ESMC,
    protein_id: str,
    sequence: str,
    logits_cfg: LogitsConfig,
):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    return client.logits(protein_tensor, logits_cfg)


def _run_batch(model: ESMC, protein_ids: Sequence[str], sequences: Sequence[str]):
    with batch_executor() as executor:
        return executor.execute_batch(
            user_func=_embed_sequence_user_func,
            client=model,
            protein_id=list(protein_ids),
            sequence=list(sequences),
            logits_cfg=_LOGITS_CFG,
        )


def _mean_pool_hidden(hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean-pool tokens per layer. Returns [num_layers, hidden_dim]. No normalization."""
    if hidden_states.dim() == 4:
        hidden_states = hidden_states.squeeze(1)
    if hidden_states.dim() != 3:
        raise ValueError(f"Unexpected hidden_states shape: {tuple(hidden_states.shape)}")
    return hidden_states.mean(dim=1)


def _postprocess(out, dtype: np.dtype) -> np.ndarray:
    """Extract mean-pooled hidden states as a numpy array."""
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("hidden_states missing from ESM output")
    return _mean_pool_hidden(hs).detach().cpu().numpy().astype(dtype, copy=False)


# -------------------------
# Parquet writer
# -------------------------

_BASE_SCHEMA = pa.schema([
    ("protein_sequence_sha256", pa.string()),
    ("sequence_length", pa.int32()),
    ("hidden_mean_blob", pa.binary()),
])


class _ParquetWriter:
    """Lazy parquet writer — opened on first write so shape dims are known."""

    def __init__(self, out_path: Path, esmc_cfg: ESMCConfig) -> None:
        self.out_path = out_path
        self.esmc_cfg = esmc_cfg
        self._writer: Optional[pq.ParquetWriter] = None

    def _open(self, hidden_layers: int, hidden_dim: int) -> None:
        _ensure_dir(self.out_path.parent)

        def _pkg_ver(name: str) -> str:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return "unknown"

        metadata = {
            b"esmc_model_name": self.esmc_cfg.model_name.encode(),
            b"esmc_np_dtype": self.esmc_cfg.np_dtype.encode(),
            b"esmc_batch_token_budget": str(self.esmc_cfg.batch_token_budget).encode(),
            b"esmc_max_single_sequence": str(self.esmc_cfg.max_single_sequence).encode(),
            b"hidden_layers": str(hidden_layers).encode(),
            b"hidden_dim": str(hidden_dim).encode(),
            b"esm_sdk_version": _pkg_ver("esm").encode(),
            b"common_core_name": b"mgnify-bgcs-common-core",
            b"common_core_version": _pkg_ver("mgnify-bgcs-common-core").encode(),
        }
        schema = _BASE_SCHEMA.with_metadata(metadata)
        self._writer = pq.ParquetWriter(str(self.out_path), schema, compression="zstd")

    def write_batch(
        self,
        protein_ids: Sequence[str],
        seq_lens: Sequence[int],
        hidden_means: Sequence[np.ndarray],
    ) -> None:
        if not protein_ids:
            return
        first = hidden_means[0]
        if self._writer is None:
            self._open(first.shape[0], first.shape[1])

        table = pa.Table.from_arrays(
            [
                pa.array(list(protein_ids), type=pa.string()),
                pa.array(list(seq_lens), type=pa.int32()),
                pa.array(
                    [memoryview(np.ascontiguousarray(h)).tobytes() for h in hidden_means],
                    type=pa.binary(),
                ),
            ],
            schema=self._writer.schema_arrow,
        )
        self._writer.write_table(table)

    def write_empty(self) -> None:
        """Write an empty parquet with metadata (dims unknown at time of writing)."""
        self._open(hidden_layers=0, hidden_dim=0)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# -------------------------
# BGC aggregation helpers
# -------------------------


def _sinusoidal_pe(n_proteins: int, dim: int) -> np.ndarray:
    """Sinusoidal positional encoding, shape (n_proteins, dim) float32.

    PE[i, 2k]   = sin(i / 10000^(2k/dim))
    PE[i, 2k+1] = cos(i / 10000^(2k/dim))

    Assumes dim is even (true for esmc_600m=1152, esmc_300m=960).
    """
    pe = np.zeros((n_proteins, dim), dtype=np.float64)
    positions = np.arange(n_proteins, dtype=np.float64)[:, np.newaxis]
    half_dim = dim // 2
    div_term = np.power(10000.0, (2.0 * np.arange(half_dim)) / dim)
    pe[:, 0::2] = np.sin(positions / div_term)
    pe[:, 1::2] = np.cos(positions / div_term)
    return pe.astype(np.float32)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def _make_bgc_strategy_id(
    layer: Union[int, Literal["final"]],
    per_protein_norm: bool,
    aggregation: str,
    post_norm: bool,
    pe_before_norm: bool,
    scale: float,
) -> str:
    return (
        f"layer_{layer}"
        f"__ppnorm_{int(per_protein_norm)}"
        f"__agg_{aggregation}"
        f"__postnorm_{int(post_norm)}"
        f"__pebefore_{int(pe_before_norm)}"
        f"__scale_{scale:.4f}"
    )


def _extract_layer_vector(
    hidden_mean: np.ndarray,
    layer: Union[int, Literal["final"]],
) -> np.ndarray:
    """Extract the (dim,) vector for a given layer from hidden_mean [n_layers, dim]."""
    if layer == "final":
        return hidden_mean[-1]
    if layer < 0 or layer >= hidden_mean.shape[0]:
        raise ValueError(f"Layer {layer} out of range (n_layers={hidden_mean.shape[0]})")
    return hidden_mean[layer]


def _aggregate_bgc_vector(
    protein_hidden_means: List[np.ndarray],
    layer: Union[int, Literal["final"]],
    scale: float,
    aggregation: Literal["mean", "max"],
    per_protein_norm: bool,
    post_norm: bool,
    pe_before_norm: bool,
) -> np.ndarray:
    """Aggregate per-protein hidden states into a single BGC vector.

    Args:
        protein_hidden_means: Ordered list of arrays, each shape [n_layers, dim].
        layer: Which hidden layer to use (int index or "final" = last layer).
        scale: PE scale factor α. 0.0 disables positional encoding.
        aggregation: "mean" or "max" pooling over proteins.
        per_protein_norm: L2-normalise each protein vector before pooling.
        post_norm: L2-normalise the BGC vector after pooling.
        pe_before_norm: If True, add α·PE before per-protein L2 norm; else after.

    Returns:
        float32 ndarray of shape (dim,).
    """
    prot_vecs = [_extract_layer_vector(h, layer) for h in protein_hidden_means]
    vecs = np.stack(prot_vecs, axis=0).astype(np.float32)  # (n, dim)
    dim = vecs.shape[1]
    n = vecs.shape[0]

    if scale == 0.0:
        if per_protein_norm:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vecs = vecs / norms
    else:
        pe = _sinusoidal_pe(n, dim)
        if pe_before_norm:
            vecs = vecs + scale * pe
            if per_protein_norm:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
        else:
            if per_protein_norm:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                vecs = vecs / norms
            vecs = vecs + scale * pe

    result = vecs.mean(axis=0) if aggregation == "mean" else vecs.max(axis=0)

    if post_norm:
        result = _l2_normalize(result)

    return result.astype(np.float32)


# -------------------------
# BGC parquet writer
# -------------------------

_BGC_SCHEMA = pa.schema([
    ("accession", pa.string()),
    ("vector_blob", pa.binary()),
    ("vector_dim", pa.int32()),
    ("strategy_id", pa.string()),
    ("n_proteins", pa.int32()),
])


class _BGCParquetWriter:
    """Lazy BGC-level parquet writer — schema metadata set on first write."""

    def __init__(self, out_path: Path, esmc_cfg: ESMCConfig, bgc_cfg: BGCAggregationConfig) -> None:
        self.out_path = out_path
        self.esmc_cfg = esmc_cfg
        self.bgc_cfg = bgc_cfg
        self._writer: Optional[pq.ParquetWriter] = None

    def _open(self, vector_dim: int, n_proteins: int) -> None:
        _ensure_dir(self.out_path.parent)

        def _pkg_ver(name: str) -> str:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return "unknown"

        metadata = {
            b"esmc_model_name": self.esmc_cfg.model_name.encode(),
            b"bgc_layer": str(self.bgc_cfg.layer).encode(),
            b"bgc_scale": f"{self.bgc_cfg.scale:.4f}".encode(),
            b"bgc_aggregation": self.bgc_cfg.aggregation.encode(),
            b"bgc_per_protein_norm": str(self.bgc_cfg.per_protein_norm).lower().encode(),
            b"bgc_post_norm": str(self.bgc_cfg.post_norm).lower().encode(),
            b"bgc_pe_before_norm": str(self.bgc_cfg.pe_before_norm).lower().encode(),
            b"vector_dim": str(vector_dim).encode(),
            b"n_proteins": str(n_proteins).encode(),
            b"esm_sdk_version": _pkg_ver("esm").encode(),
            b"common_core_name": b"mgnify-bgcs-common-core",
            b"common_core_version": _pkg_ver("mgnify-bgcs-common-core").encode(),
        }
        schema = _BGC_SCHEMA.with_metadata(metadata)
        self._writer = pq.ParquetWriter(str(self.out_path), schema, compression="zstd")

    def write(
        self,
        accession: Optional[str],
        vector: np.ndarray,
        strategy_id: str,
        n_proteins: int,
    ) -> None:
        if self._writer is None:
            self._open(int(vector.shape[0]), n_proteins)
        blob = memoryview(np.ascontiguousarray(vector)).tobytes()
        table = pa.Table.from_arrays(
            [
                pa.array([accession], type=pa.string()),
                pa.array([blob], type=pa.binary()),
                pa.array([int(vector.shape[0])], type=pa.int32()),
                pa.array([strategy_id], type=pa.string()),
                pa.array([n_proteins], type=pa.int32()),
            ],
            schema=self._writer.schema_arrow,
        )
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


# -------------------------
# Core runner
# -------------------------


def embed_and_write_parquet(cfg: JobSettings) -> None:
    device = _resolve_device(cfg.esmc.device)
    model = _get_model(cfg.esmc.model_name, device)
    dtype = _np_dtype(cfg.esmc.np_dtype)

    out_path = _ensure_dir(cfg.io.output_dir) / f"{cfg.io.output_prefix}.embeddings.parquet"
    writer = _ParquetWriter(out_path, cfg.esmc)

    if cfg.esmc.validate_sequence:
        validate_input_file(cfg.io.input_file, cfg.io.input_format)

    total_seen = total_written = total_failed = total_retried = 0

    try:
        batches = _batch_by_budget(
            iter_proteins(cfg.io.input_file, cfg.io.input_format),
            budget=cfg.esmc.batch_token_budget,
            max_single_sequence=cfg.esmc.max_single_sequence,
        )

        for batch in tqdm(batches, desc="Embedding"):
            pids = [r.protein_id for r in batch]
            seqs = [r.sequence for r in batch]
            slens = [len(s) for s in seqs]

            if cfg.esmc.validate_sequence:
                for pid, seq in zip(pids, seqs):
                    validate_protein_sequence(pid, seq)

            outputs = _run_batch(model, pids, seqs)

            retry_records: List[_ProteinRecord] = []
            out_pids: List[str] = []
            out_lens: List[int] = []
            hid_rows: List[np.ndarray] = []

            def _collect(pid: str, slen: int, out_obj) -> None:
                hid_rows.append(_postprocess(out_obj, dtype))
                out_pids.append(pid)
                out_lens.append(slen)

            for pid, seq, slen, out in zip(pids, seqs, slens, outputs):
                total_seen += 1

                if isinstance(out, BaseException):
                    total_failed += 1
                    log.warning("Batch failure pid=%s len=%d err=%r", pid, slen, out)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise RuntimeError(f"ESM failed pid={pid}") from out
                    continue

                if getattr(out, "hidden_states", None) is None:
                    total_failed += 1
                    log.warning("Missing hidden_states pid=%s len=%d", pid, slen)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise RuntimeError(f"Missing hidden_states pid={pid}")
                    continue

                try:
                    _collect(pid, slen, out)
                except Exception as e:
                    total_failed += 1
                    log.warning("Postprocess failure pid=%s len=%d err=%r", pid, slen, e)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise

            for r in retry_records:
                total_retried += 1
                try:
                    retry_outs = _run_batch(model, [r.protein_id], [r.sequence])
                    retry_out = retry_outs[0]
                    if isinstance(retry_out, BaseException):
                        raise retry_out
                    if getattr(retry_out, "hidden_states", None) is None:
                        raise RuntimeError("Missing hidden_states on retry")
                    _collect(r.protein_id, len(r.sequence), retry_out)
                except Exception as e:
                    total_failed += 1
                    log.error(
                        "Retry failed pid=%s len=%d err=%r seq_prefix=%s",
                        r.protein_id, len(r.sequence), e, r.sequence[:50],
                    )
                    if not cfg.esmc.skip_failed_records:
                        raise

            if out_pids:
                writer.write_batch(out_pids, out_lens, hid_rows)
                total_written += len(out_pids)

        if total_seen == 0:
            log.info("No sequences found — writing empty parquet: %s", out_path)
            writer.write_empty()
        elif total_written == 0:
            raise RuntimeError("No successful embeddings were produced")

        log.info(
            "wrote parquet: %s written=%d seen=%d failed=%d retried=%d",
            out_path, total_written, total_seen, total_failed, total_retried,
        )

    finally:
        writer.close()


# -------------------------
# BGC core runner
# -------------------------


def embed_and_write_bgc_parquet(cfg: BGCJobSettings) -> None:
    """Embed all proteins in the input file and write a single-row BGC embedding parquet.

    Proteins are taken in the order they appear in the input file (CDS order for GenBank).
    Optionally also writes a per-protein parquet (cfg.bgc.write_protein_parquet).
    """
    device = _resolve_device(cfg.esmc.device)
    model = _get_model(cfg.esmc.model_name, device)
    dtype = _np_dtype(cfg.esmc.np_dtype)

    out_dir = _ensure_dir(cfg.io.output_dir)
    bgc_out_path = out_dir / f"{cfg.bgc.output_prefix}.embedding.parquet"
    bgc_writer = _BGCParquetWriter(bgc_out_path, cfg.esmc, cfg.bgc)

    protein_writer: Optional[_ParquetWriter] = None
    if cfg.bgc.write_protein_parquet:
        prot_out_path = out_dir / f"{cfg.io.output_prefix}.embeddings.parquet"
        protein_writer = _ParquetWriter(prot_out_path, cfg.esmc)

    if cfg.esmc.validate_sequence:
        validate_input_file(cfg.io.input_file, cfg.io.input_format)

    # Collect all proteins in order; preserve ordering via OrderedDict (first-seen wins).
    ordered_pids: List[str] = []
    ordered_seqs: List[str] = []
    seen_pids: set = set()
    for pid, seq in iter_proteins(cfg.io.input_file, cfg.io.input_format):
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        ordered_pids.append(pid)
        ordered_seqs.append(seq)

    if not ordered_pids:
        raise ValueError(f"No protein sequences found in {cfg.io.input_file}")

    # Embed all proteins; results keyed by pid.
    results: Dict[str, Optional[np.ndarray]] = {pid: None for pid in ordered_pids}
    total_seen = total_written_prot = total_failed = total_retried = 0

    try:
        for batch in tqdm(
            _batch_by_budget(
                iter(zip(ordered_pids, ordered_seqs)),
                budget=cfg.esmc.batch_token_budget,
                max_single_sequence=cfg.esmc.max_single_sequence,
            ),
            desc="Embedding BGC proteins",
        ):
            pids = [r.protein_id for r in batch]
            seqs = [r.sequence for r in batch]
            slens = [len(s) for s in seqs]

            if cfg.esmc.validate_sequence:
                for pid, seq in zip(pids, seqs):
                    validate_protein_sequence(pid, seq)

            outputs = _run_batch(model, pids, seqs)

            retry_records: List[_ProteinRecord] = []
            out_pids: List[str] = []
            out_lens: List[int] = []
            hid_rows: List[np.ndarray] = []

            def _collect(pid: str, slen: int, out_obj) -> None:
                hid = _postprocess(out_obj, dtype)
                results[pid] = hid
                hid_rows.append(hid)
                out_pids.append(pid)
                out_lens.append(slen)

            for pid, seq, slen, out in zip(pids, seqs, slens, outputs):
                total_seen += 1
                if isinstance(out, BaseException):
                    total_failed += 1
                    log.warning("Batch failure pid=%s len=%d err=%r", pid, slen, out)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise RuntimeError(f"ESM failed pid={pid}") from out
                    continue
                if getattr(out, "hidden_states", None) is None:
                    total_failed += 1
                    log.warning("Missing hidden_states pid=%s len=%d", pid, slen)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise RuntimeError(f"Missing hidden_states pid={pid}")
                    continue
                try:
                    _collect(pid, slen, out)
                except Exception as e:
                    total_failed += 1
                    log.warning("Postprocess failure pid=%s len=%d err=%r", pid, slen, e)
                    if cfg.esmc.retry_failed_individually:
                        retry_records.append(_ProteinRecord(pid, seq))
                    elif not cfg.esmc.skip_failed_records:
                        raise

            for r in retry_records:
                total_retried += 1
                try:
                    retry_outs = _run_batch(model, [r.protein_id], [r.sequence])
                    retry_out = retry_outs[0]
                    if isinstance(retry_out, BaseException):
                        raise retry_out
                    if getattr(retry_out, "hidden_states", None) is None:
                        raise RuntimeError("Missing hidden_states on retry")
                    _collect(r.protein_id, len(r.sequence), retry_out)
                except Exception as e:
                    total_failed += 1
                    log.error(
                        "Retry failed pid=%s len=%d err=%r",
                        r.protein_id, len(r.sequence), e,
                    )
                    if not cfg.esmc.skip_failed_records:
                        raise

            if out_pids and protein_writer is not None:
                protein_writer.write_batch(out_pids, out_lens, hid_rows)
                total_written_prot += len(out_pids)

        # Build ordered list of successfully-embedded hidden means.
        protein_hidden_means: List[np.ndarray] = []
        for pid in ordered_pids:
            hm = results.get(pid)
            if hm is None:
                raise RuntimeError(
                    f"Protein pid={pid} failed embedding; cannot produce BGC vector. "
                    "Set skip_failed_records=false to surface individual errors."
                )
            protein_hidden_means.append(hm)

        bgc_vec = _aggregate_bgc_vector(
            protein_hidden_means,
            layer=cfg.bgc.layer,
            scale=cfg.bgc.scale,
            aggregation=cfg.bgc.aggregation,
            per_protein_norm=cfg.bgc.per_protein_norm,
            post_norm=cfg.bgc.post_norm,
            pe_before_norm=cfg.bgc.pe_before_norm,
        )
        strategy_id = _make_bgc_strategy_id(
            cfg.bgc.layer, cfg.bgc.per_protein_norm, cfg.bgc.aggregation,
            cfg.bgc.post_norm, cfg.bgc.pe_before_norm, cfg.bgc.scale,
        )
        bgc_writer.write(cfg.bgc.accession, bgc_vec, strategy_id, len(protein_hidden_means))

        log.info(
            "wrote BGC parquet: %s  dim=%d  n_proteins=%d  strategy=%s",
            bgc_out_path, int(bgc_vec.shape[0]), len(protein_hidden_means), strategy_id,
        )
        if protein_writer is not None:
            log.info(
                "wrote protein parquet: %s  written=%d  seen=%d  failed=%d  retried=%d",
                protein_writer.out_path, total_written_prot, total_seen,
                total_failed, total_retried,
            )

    finally:
        bgc_writer.close()
        if protein_writer is not None:
            protein_writer.close()


# -------------------------
# Library API
# -------------------------


def embed_sequences(
    sequences: Sequence[str],
    *,
    model_name: str = "esmc_600m",
    device: Optional[str] = None,
    np_dtype: str = "float32",
    batch_token_budget: int = 6000,
    max_single_sequence: int = 10000,
) -> list[np.ndarray]:
    """Embed protein sequences using ESM-C.

    Returns one ``np.ndarray`` of shape ``[num_layers, hidden_dim]`` per
    input sequence.  No normalization is applied.  The model is cached
    globally after the first call.

    Args:
        sequences: Protein sequences (plain amino-acid strings).
        model_name: ESM-C checkpoint name (default ``esmc_600m``).
        device: ``"cpu"`` or ``"cuda"``.  Auto-detected when ``None``.
        np_dtype: ``"float32"`` or ``"float16"``.
        batch_token_budget: Max total tokens per inference batch.
        max_single_sequence: Sequences longer than this are skipped.

    Returns:
        List of arrays, one per input sequence, in input order.
        Sequences that failed inference are returned as ``None``.
    """
    resolved_device = _resolve_device(device)
    model = _get_model(model_name, resolved_device)
    dtype = _np_dtype(np_dtype)

    id_seq: list[tuple[str, str]] = [
        (str(i), seq) for i, seq in enumerate(sequences)
    ]
    results: dict[str, Optional[np.ndarray]] = {str(i): None for i in range(len(sequences))}

    for batch in _batch_by_budget(
        iter(id_seq),
        budget=batch_token_budget,
        max_single_sequence=max_single_sequence,
    ):
        pids = [r.protein_id for r in batch]
        seqs = [r.sequence for r in batch]
        outputs = _run_batch(model, pids, seqs)

        for pid, out in zip(pids, outputs):
            if isinstance(out, BaseException) or getattr(out, "hidden_states", None) is None:
                log.warning("embed_sequences: failed for index=%s err=%r", pid, out)
                continue
            try:
                results[pid] = _postprocess(out, dtype)
            except Exception as e:
                log.warning("embed_sequences: postprocess failed index=%s err=%r", pid, e)

    return [results[str(i)] for i in range(len(sequences))]


def aggregate_bgc_sequences(
    sequences: Sequence[str],
    *,
    model_name: str = "esmc_600m",
    layer: Union[int, Literal["final"]] = 29,
    scale: float = 0.5,
    aggregation: Literal["mean", "max"] = "mean",
    per_protein_norm: bool = False,
    post_norm: bool = False,
    pe_before_norm: bool = True,
    device: Optional[str] = None,
    np_dtype: str = "float32",
    batch_token_budget: int = 6000,
    max_single_sequence: int = 10000,
) -> Optional[np.ndarray]:
    """Embed protein sequences and aggregate into a single BGC-level vector.

    Sequences must be provided in CDS order.  Sinusoidal positional encoding
    is applied at the specified scale before mean/max pooling.

    Args:
        sequences: Protein sequences in CDS order.
        layer: Hidden layer index (int) or ``"final"`` (last hidden layer).
        scale: PE scale factor α.  0.0 disables positional encoding.
        aggregation: ``"mean"`` or ``"max"`` pooling over proteins.
        per_protein_norm: L2-normalise each protein vector before pooling.
        post_norm: L2-normalise the BGC vector after pooling.
        pe_before_norm: If True, add α·PE before per-protein L2 norm; else after.
        model_name: ESM-C checkpoint (default ``esmc_600m``).
        device: ``"cpu"`` or ``"cuda"``.  Auto-detected when ``None``.
        np_dtype: ``"float32"`` or ``"float16"``.
        batch_token_budget: Max total tokens per inference batch.
        max_single_sequence: Sequences longer than this are skipped.

    Returns:
        float32 ndarray of shape ``(hidden_dim,)``, or ``None`` if any
        embedding failed.
    """
    hidden_means = embed_sequences(
        sequences,
        model_name=model_name,
        device=device,
        np_dtype=np_dtype,
        batch_token_budget=batch_token_budget,
        max_single_sequence=max_single_sequence,
    )
    if any(h is None for h in hidden_means):
        log.warning(
            "aggregate_bgc_sequences: %d/%d sequences failed embedding; returning None",
            sum(h is None for h in hidden_means),
            len(hidden_means),
        )
        return None
    return _aggregate_bgc_vector(
        hidden_means,  # type: ignore[arg-type]
        layer=layer,
        scale=scale,
        aggregation=aggregation,
        per_protein_norm=per_protein_norm,
        post_norm=post_norm,
        pe_before_norm=pe_before_norm,
    )


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed proteins with ESM-C and write to parquet.")
    p.add_argument("--config", help="Path to YAML config file")
    p.add_argument("--job-name")

    p.add_argument("--input-file")
    p.add_argument("--format", choices=["fasta", "genbank"])
    p.add_argument("--output-dir")
    p.add_argument("--output-prefix")

    p.add_argument("--model-name")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--np-dtype", choices=["float32", "float16"])

    p.add_argument("--batch-token-budget", type=int)
    p.add_argument("--max-single-sequence", type=int)

    p.add_argument("--no-validate-sequence", action="store_true")
    p.add_argument("--no-retry-failed-individually", action="store_true")
    p.add_argument("--skip-failed-records", action="store_true")

    p.add_argument("--log-level")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict = {}

    if args.job_name:
        overrides["job_name"] = args.job_name

    if args.input_file:
        overrides.setdefault("io", {})["input_file"] = args.input_file
    if args.format:
        overrides.setdefault("io", {})["input_format"] = args.format
    if args.output_dir:
        overrides.setdefault("io", {})["output_dir"] = args.output_dir
    if args.output_prefix:
        overrides.setdefault("io", {})["output_prefix"] = args.output_prefix

    if args.model_name:
        overrides.setdefault("esmc", {})["model_name"] = args.model_name
    if args.device:
        overrides.setdefault("esmc", {})["device"] = args.device
    if args.np_dtype:
        overrides.setdefault("esmc", {})["np_dtype"] = args.np_dtype
    if args.batch_token_budget is not None:
        overrides.setdefault("esmc", {})["batch_token_budget"] = args.batch_token_budget
    if args.max_single_sequence is not None:
        overrides.setdefault("esmc", {})["max_single_sequence"] = args.max_single_sequence
    if args.no_validate_sequence:
        overrides.setdefault("esmc", {})["validate_sequence"] = False
    if args.no_retry_failed_individually:
        overrides.setdefault("esmc", {})["retry_failed_individually"] = False
    if args.skip_failed_records:
        overrides.setdefault("esmc", {})["skip_failed_records"] = True

    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level

    cfg = load_settings(
        JobSettings,
        yaml_path=args.config,
        cli_overrides=overrides,
        options=LoaderOptions(env_prefix="ESMJOB_"),
    )

    setup_logging(cfg.logging)

    log.info("running %s version %s", cfg.job_name, dist_version(__name__))
    log.info("config: %s", cfg.model_dump())

    embed_and_write_parquet(cfg)


if __name__ == "__main__":
    main()


# -------------------------
# BGC CLI
# -------------------------


def parse_bgc_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Embed proteins with ESM-C and aggregate into a BGC-level vector "
            "using sinusoidal positional encoding."
        )
    )
    p.add_argument("--config", help="Path to YAML config file")
    p.add_argument("--job-name")

    p.add_argument("--input-file")
    p.add_argument("--format", choices=["fasta", "genbank"])
    p.add_argument("--output-dir")
    p.add_argument("--output-prefix", help="Per-protein parquet prefix (default: proteins)")

    p.add_argument("--model-name")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--np-dtype", choices=["float32", "float16"])
    p.add_argument("--batch-token-budget", type=int)
    p.add_argument("--max-single-sequence", type=int)
    p.add_argument("--no-validate-sequence", action="store_true")
    p.add_argument("--no-retry-failed-individually", action="store_true")
    p.add_argument("--skip-failed-records", action="store_true")

    p.add_argument(
        "--layer",
        help='Hidden layer index (int) or "final" (last layer). Default: 29.',
    )
    p.add_argument("--scale", type=float, help="PE scale α. Default: 0.5.")
    p.add_argument("--aggregation", choices=["mean", "max"])
    p.add_argument(
        "--per-protein-norm",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        metavar="BOOL",
    )
    p.add_argument(
        "--post-norm",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        metavar="BOOL",
    )
    p.add_argument(
        "--pe-before-norm",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        metavar="BOOL",
    )
    p.add_argument(
        "--no-write-protein-parquet",
        action="store_true",
        help="Skip writing the per-protein embeddings parquet.",
    )
    p.add_argument("--accession", help="BGC accession written to the output row.")
    p.add_argument("--bgc-output-prefix", help="BGC parquet filename prefix (default: bgc)")

    p.add_argument("--log-level")
    return p.parse_args()


def main_bgc() -> None:
    args = parse_bgc_args()
    overrides: dict = {}

    if args.job_name:
        overrides["job_name"] = args.job_name

    if args.input_file:
        overrides.setdefault("io", {})["input_file"] = args.input_file
    if args.format:
        overrides.setdefault("io", {})["input_format"] = args.format
    if args.output_dir:
        overrides.setdefault("io", {})["output_dir"] = args.output_dir
    if args.output_prefix:
        overrides.setdefault("io", {})["output_prefix"] = args.output_prefix

    if args.model_name:
        overrides.setdefault("esmc", {})["model_name"] = args.model_name
    if args.device:
        overrides.setdefault("esmc", {})["device"] = args.device
    if args.np_dtype:
        overrides.setdefault("esmc", {})["np_dtype"] = args.np_dtype
    if args.batch_token_budget is not None:
        overrides.setdefault("esmc", {})["batch_token_budget"] = args.batch_token_budget
    if args.max_single_sequence is not None:
        overrides.setdefault("esmc", {})["max_single_sequence"] = args.max_single_sequence
    if args.no_validate_sequence:
        overrides.setdefault("esmc", {})["validate_sequence"] = False
    if args.no_retry_failed_individually:
        overrides.setdefault("esmc", {})["retry_failed_individually"] = False
    if args.skip_failed_records:
        overrides.setdefault("esmc", {})["skip_failed_records"] = True

    if args.layer is not None:
        raw = args.layer
        overrides.setdefault("bgc", {})["layer"] = raw if raw == "final" else int(raw)
    if args.scale is not None:
        overrides.setdefault("bgc", {})["scale"] = args.scale
    if args.aggregation:
        overrides.setdefault("bgc", {})["aggregation"] = args.aggregation
    if args.per_protein_norm is not None:
        overrides.setdefault("bgc", {})["per_protein_norm"] = args.per_protein_norm
    if args.post_norm is not None:
        overrides.setdefault("bgc", {})["post_norm"] = args.post_norm
    if args.pe_before_norm is not None:
        overrides.setdefault("bgc", {})["pe_before_norm"] = args.pe_before_norm
    if args.no_write_protein_parquet:
        overrides.setdefault("bgc", {})["write_protein_parquet"] = False
    if args.accession:
        overrides.setdefault("bgc", {})["accession"] = args.accession
    if args.bgc_output_prefix:
        overrides.setdefault("bgc", {})["output_prefix"] = args.bgc_output_prefix

    if args.log_level:
        overrides.setdefault("logging", {})["level"] = args.log_level

    cfg = load_settings(
        BGCJobSettings,
        yaml_path=args.config,
        cli_overrides=overrides,
        options=LoaderOptions(env_prefix="ESMJOB_BGC_"),
    )

    setup_logging(cfg.logging)

    log.info("running %s version %s", cfg.job_name, dist_version(__name__))
    log.info("config: %s", cfg.model_dump())

    embed_and_write_bgc_parquet(cfg)
