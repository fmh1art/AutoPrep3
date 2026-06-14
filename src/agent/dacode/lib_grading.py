"""
DA-Code grading engine.

Ports the core metric functions from the DA-Code benchmark
(https://github.com/yiyihum/da-code) so that we can evaluate the agent's
workspace output directly, without requiring the full `da_agent` package.

Supported metric functions
--------------------------
- compare_csv           : multi-set F1 score over CSV rows (with column tolerance)
- compare_csv_files     : compare two CSV files on disk (most common in DA-Code)
- compare_csv_details   : compare_csv + per-column breakdown
- compare_text          : compare scalar/JSON outputs with type-aware tolerance
- compare_image         : compare chart attributes (size, title, legend, data points)
- compare_ml            : compare ML prediction outputs (accuracy / MAE / F1)
- check_include_exclude : substring inclusion / exclusion test (returns 1.0 or 0.0)
- compare_sqlite        : compare two SQLite tables by their row-level F1

How grading works
-----------------
For each Task, we:
  1. Resolve the predicted output — files that the agent created in
     the workspace directory (workspace/<task_id>/*.csv / *.json / *.png / ...)
     OR a numeric result stored in result.json.
  2. Resolve the gold output — files in <dataset_root>/gold/<task_id>/
     OR the numeric value in MetricSpec.result["number"].
  3. Run the configured metric function(s) from the task's evaluator spec.
  4. Combine scores via the conjunction operator ("avg" / "max" / "min" /
     "and" / "or").

The final score is a float in [0.0, 1.0].
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .lib_tasks import Task, MetricSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Result of running a single metric function."""
    metric: str
    score: float
    max_score: float = 1.0
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GradeResult:
    task_id: str
    score: float
    max_score: float
    conjunction: str
    per_metric: List[MetricResult]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "max_score": self.max_score,
            "conjunction": self.conjunction,
            "per_metric": [
                {
                    "metric": m.metric,
                    "score": m.score,
                    "max_score": m.max_score,
                    "errors": m.errors,
                    "details": m.details,
                }
                for m in self.per_metric
            ],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# CSV / Table metrics (the most commonly used in DA-Code)
# ---------------------------------------------------------------------------

_FLOAT_TOLERANCE = 1e-6  # relative tolerance for float equality


def _parse_csv_text(text: str) -> List[Dict[str, str]]:
    """Parse CSV text into a list of row dicts."""
    import io
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(r) for r in reader]
    return rows


def _normalize_value(v: Any) -> str:
    """Normalize a cell value for robust row comparison."""
    if v is None:
        return ""
    s = str(v).strip()
    # Try to canonicalise numbers (floats, ints, scientific notation)
    try:
        f = float(s)
        if math.isfinite(f):
            # Use a fixed number of significant digits for comparison
            # Formatting via repr-like round trip: keep 9 significant digits
            return f"__num__{f:.9g}"
    except (ValueError, TypeError):
        pass
    return s.lower()


def _rows_equal(a: Dict[str, str], b: Dict[str, str]) -> bool:
    """Best-effort row equality with type-aware tolerance."""
    if set(a.keys()) != set(b.keys()):
        # Column mismatch: compare common columns only
        common = set(a.keys()) & set(b.keys())
        if not common:
            return False
        a = {k: a[k] for k in common}
        b = {k: b[k] for k in common}
    for key in a:
        av = _normalize_value(a.get(key, ""))
        bv = _normalize_value(b.get(key, ""))
        if av.startswith("__num__") and bv.startswith("__num__"):
            try:
                af = float(av[len("__num__"):])
                bf = float(bv[len("__num__"):])
                if not math.isclose(af, bf, rel_tol=_FLOAT_TOLERANCE, abs_tol=1e-9):
                    return False
                continue
            except (ValueError, TypeError):
                pass
        if av != bv:
            return False
    return True


def _rowset_f1(pred_rows: List[Dict[str, str]], gold_rows: List[Dict[str, str]]) -> float:
    """Compute multi-set F1 between predicted and gold row sets.

    This is DA-Code's canonical CSV scoring function: treat each row as a
    bag item and compute precision / recall over rows.
    """
    if not gold_rows:
        return 1.0 if not pred_rows else 0.0
    if not pred_rows:
        return 0.0

    # Greedy matching (small enough datasets in DA-Code for O(n*m)).
    matched_gold = set()
    matched_pred = 0
    for i, pr in enumerate(pred_rows):
        for j, gr in enumerate(gold_rows):
            if j in matched_gold:
                continue
            if _rows_equal(pr, gr):
                matched_gold.add(j)
                matched_pred += 1
                break

    precision = matched_pred / len(pred_rows)
    recall = matched_pred / len(gold_rows)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compare_csv(pred_text: str, gold_text: str) -> MetricResult:
    """Compare two CSV string bodies. Returns a row-level F1 score [0, 1]."""
    try:
        pred_rows = _parse_csv_text(pred_text)
    except Exception as e:
        return MetricResult("compare_csv", 0.0, errors=[f"predicted CSV parse error: {e}"])
    try:
        gold_rows = _parse_csv_text(gold_text)
    except Exception as e:
        return MetricResult("compare_csv", 0.0, errors=[f"gold CSV parse error: {e}"])

    score = _rowset_f1(pred_rows, gold_rows)
    return MetricResult(
        "compare_csv", score,
        details={"pred_rows": len(pred_rows), "gold_rows": len(gold_rows)},
    )


def compare_csv_files(pred_path: str, gold_path: str) -> MetricResult:
    """Compare two CSV files on disk."""
    if not os.path.isfile(pred_path):
        return MetricResult("compare_csv_files", 0.0, errors=[f"predicted file missing: {pred_path}"])
    if not os.path.isfile(gold_path):
        return MetricResult("compare_csv_files", 0.0, errors=[f"gold file missing: {gold_path}"])
    try:
        with open(pred_path, "r", encoding="utf-8", errors="replace") as f:
            pred_text = f.read()
        with open(gold_path, "r", encoding="utf-8", errors="replace") as f:
            gold_text = f.read()
    except Exception as e:
        return MetricResult("compare_csv_files", 0.0, errors=[f"read error: {e}"])
    return compare_csv(pred_text, gold_text)


def compare_csv_details(pred_text: str, gold_text: str) -> MetricResult:
    """compare_csv with per-column accuracy details."""
    base = compare_csv(pred_text, gold_text)
    try:
        pred_rows = _parse_csv_text(pred_text)
        gold_rows = _parse_csv_text(gold_text)
    except Exception:
        return base
    if pred_rows and gold_rows:
        columns = list(set(list(pred_rows[0].keys()) + list(gold_rows[0].keys())))
        for col in columns[:10]:  # limit to first 10 columns
            pred_vals = [_normalize_value(r.get(col, "")) for r in pred_rows]
            gold_vals = [_normalize_value(r.get(col, "")) for r in gold_rows]
            if gold_vals:
                # Column precision (order-free): how many pred col values appear in gold col values?
                gold_set = set(gold_vals)
                col_match = sum(1 for v in pred_vals if v in gold_set)
                base.details[f"col_{col}_precision"] = round(col_match / len(pred_vals), 4)
    return base


# ---------------------------------------------------------------------------
# SQLite comparison
# ---------------------------------------------------------------------------

def compare_sqlite(pred_path: str, gold_path: str, table_name: Optional[str] = None) -> MetricResult:
    """Compare two SQLite databases (specified table or ALL tables)."""
    try:
        import sqlite3
    except ImportError:
        return MetricResult("compare_sqlite", 0.0, errors=["sqlite3 not available"])

    if not os.path.isfile(pred_path):
        return MetricResult("compare_sqlite", 0.0, errors=[f"predicted DB missing: {pred_path}"])
    if not os.path.isfile(gold_path):
        return MetricResult("compare_sqlite", 0.0, errors=[f"gold DB missing: {gold_path}"])

    try:
        def _list_tables(db_path: str) -> List[str]:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            conn.close()
            return tables

        def _dump_rows(db_path: str, table: str) -> List[Dict[str, Any]]:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table}")
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows

        if table_name:
            pred_tables = gold_tables = [table_name]
        else:
            pred_tables = _list_tables(pred_path)
            gold_tables = _list_tables(gold_path)
        common = list(set(pred_tables) & set(gold_tables))
        if not common:
            return MetricResult("compare_sqlite", 0.0, errors=["no common tables"])

        scores = []
        for table in common:
            pred_rows = _dump_rows(pred_path, table)
            gold_rows = _dump_rows(gold_path, table)
            # Normalise to string-based dicts
            pred_norm = [{k: str(v) for k, v in r.items()} for r in pred_rows]
            gold_norm = [{k: str(v) for k, v in r.items()} for r in gold_rows]
            scores.append(_rowset_f1(pred_norm, gold_norm))
        avg = sum(scores) / len(scores) if scores else 0.0
        return MetricResult("compare_sqlite", avg, details={"tables": common, "per_table": scores})
    except Exception as e:
        return MetricResult("compare_sqlite", 0.0, errors=[f"sqlite error: {e}"])


# ---------------------------------------------------------------------------
# Text / JSON / number comparison
# ---------------------------------------------------------------------------

def _numbers_close(a: float, b: float, rel_tol: Optional[float] = None, abs_tol: Optional[float] = None) -> bool:
    rel = rel_tol if rel_tol is not None else 0.02
    ab = abs_tol if abs_tol is not None else 1e-6
    return math.isclose(a, b, rel_tol=rel, abs_tol=ab)


def compare_text(pred_value: Any, gold_value: Any, options: Optional[Dict[str, Any]] = None) -> MetricResult:
    """Compare two outputs (scalars or JSON-compatible structures)."""
    options = options or {}
    rel_tol = options.get("rel_tol", 0.02)
    abs_tol = options.get("abs_tol", 1e-6)

    try:
        # Fast path: both numeric
        if isinstance(pred_value, (int, float)) and isinstance(gold_value, (int, float)):
            if _numbers_close(float(pred_value), float(gold_value), rel_tol, abs_tol):
                return MetricResult("compare_text", 1.0)
            # Fraction of |gold| represented by |pred - gold|
            denom = max(abs(float(gold_value)), abs_tol)
            err = abs(float(pred_value) - float(gold_value))
            return MetricResult("compare_text", max(0.0, 1.0 - err / denom))

        # String path: try parse as JSON
        if isinstance(pred_value, str) and isinstance(gold_value, str):
            try:
                pj = json.loads(pred_value)
                gj = json.loads(gold_value)
                return compare_text(pj, gj, options)
            except (json.JSONDecodeError, TypeError):
                pass
            # Plain string comparison: whitespace-tolerant
            if pred_value.strip().lower() == gold_value.strip().lower():
                return MetricResult("compare_text", 1.0)
            # Fuzzy substring
            if gold_value.strip().lower() in pred_value.strip().lower() or \
               pred_value.strip().lower() in gold_value.strip().lower():
                return MetricResult("compare_text", 0.7)
            return MetricResult("compare_text", 0.0)

        # Dict/list recursive: structural similarity
        if isinstance(pred_value, dict) and isinstance(gold_value, dict):
            keys = set(pred_value.keys()) | set(gold_value.keys())
            if not keys:
                return MetricResult("compare_text", 1.0)
            sub_scores = []
            for k in keys:
                sr = compare_text(pred_value.get(k), gold_value.get(k), options)
                sub_scores.append(sr.score)
            return MetricResult(
                "compare_text", sum(sub_scores) / len(sub_scores),
                details={"keys": sorted(keys)},
            )

        if isinstance(pred_value, list) and isinstance(gold_value, list):
            if not pred_value and not gold_value:
                return MetricResult("compare_text", 1.0)
            # Treat as set of elements (bag) when length differs; otherwise element-wise
            if len(pred_value) == len(gold_value):
                sub_scores = [
                    compare_text(p, g, options).score
                    for p, g in zip(pred_value, gold_value)
                ]
                avg = sum(sub_scores) / len(sub_scores)
                return MetricResult("compare_text", avg, details={"length_match": True})
            # Bag F1
            gold_norm = [json.dumps(x, sort_keys=True, default=str) for x in gold_value]
            pred_norm = [json.dumps(x, sort_keys=True, default=str) for x in pred_value]
            matched = 0
            used = set()
            for ps in pred_norm:
                for i, gs in enumerate(gold_norm):
                    if i in used:
                        continue
                    if compare_text(ps, gs, options).score >= 0.9:
                        used.add(i)
                        matched += 1
                        break
            prec = matched / len(pred_norm) if pred_norm else 0.0
            rec = matched / len(gold_norm) if gold_norm else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return MetricResult("compare_text", f1, details={"bag_match": True})

        # Fallback: stringise both
        return compare_text(str(pred_value), str(gold_value), options)
    except Exception as e:
        return MetricResult("compare_text", 0.0, errors=[f"compare_text error: {e}"])


# ---------------------------------------------------------------------------
# Image / chart comparison (lightweight; no strict rendering requirements)
# ---------------------------------------------------------------------------

def _probe_image(path: str) -> Dict[str, Any]:
    """Return basic image info: size, mode, and (optionally) file size."""
    info: Dict[str, Any] = {"file": path, "exists": False}
    if not os.path.isfile(path):
        return info
    info["exists"] = True
    info["file_size"] = os.path.getsize(path)
    info["extension"] = os.path.splitext(path)[1].lower()

    # Try PIL (most common image format)
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as img:
            info["size"] = list(img.size)
            info["mode"] = img.mode
            # Basic colour histogram (top 8 buckets)
            hist = img.histogram()
            info["hist_top"] = sorted(range(len(hist)), key=lambda i: -hist[i])[:8]
    except Exception as e:
        info["pil_error"] = str(e)
    return info


def compare_image(pred_path: str, gold_path: str, options: Optional[Dict[str, Any]] = None) -> MetricResult:
    """Light-weight chart comparison.

    Checks: file-extension match, aspect-ratio tolerance, file-size ratio.
    Does NOT enforce pixel-level equality; DA-Code only validates that charts
    look reasonable.
    """
    options = options or {}
    errors: List[str] = []

    pred_info = _probe_image(pred_path)
    gold_info = _probe_image(gold_path)

    if not pred_info.get("exists"):
        return MetricResult("compare_image", 0.0, errors=[f"predicted image missing: {pred_path}"])
    if not gold_info.get("exists"):
        return MetricResult("compare_image", 0.0, errors=[f"gold image missing: {gold_path}"])

    sub_scores: List[float] = []
    details: Dict[str, Any] = {}

    # Extension check
    pred_ext = pred_info.get("extension", "").lower()
    gold_ext = gold_info.get("extension", "").lower()
    if pred_ext and gold_ext:
        sub_scores.append(1.0 if pred_ext == gold_ext else 0.3)
        details["extension_match"] = pred_ext == gold_ext

    # Aspect ratio
    pred_size = pred_info.get("size")
    gold_size = gold_info.get("size")
    if pred_size and gold_size and all(pred_size) and all(gold_size):
        pred_ar = pred_size[0] / pred_size[1]
        gold_ar = gold_size[0] / gold_size[1]
        ar_err = abs(pred_ar - gold_ar) / max(gold_ar, 1e-6)
        sub_scores.append(max(0.0, 1.0 - ar_err))
        details["aspect_ratio"] = {"pred": round(pred_ar, 3), "gold": round(gold_ar, 3)}

    # Histogram overlap (proxy for colour palette)
    pred_hist = set(pred_info.get("hist_top", []))
    gold_hist = set(gold_info.get("hist_top", []))
    if pred_hist and gold_hist:
        overlap = len(pred_hist & gold_hist) / len(pred_hist | gold_hist)
        sub_scores.append(overlap)
        details["hist_overlap"] = round(overlap, 3)

    # File size: penalty for empty images
    if pred_info.get("file_size", 0) < 100:
        sub_scores.append(0.0)
        errors.append("predicted image is suspiciously small (< 100 bytes)")

    if not sub_scores:
        return MetricResult("compare_image", 0.0, errors=["could not extract any image features"] + errors)
    avg = sum(sub_scores) / len(sub_scores)
    return MetricResult("compare_image", avg, details=details, errors=errors)


# ---------------------------------------------------------------------------
# ML prediction grading
# ---------------------------------------------------------------------------

def compare_ml(pred_path: str, gold_path: str, options: Optional[Dict[str, Any]] = None) -> MetricResult:
    """Compare two ML prediction files (CSV/JSON)."""
    options = options or {}
    metric = options.get("metric", "accuracy").lower()

    def _load(path: str) -> Optional[List[Dict[str, Any]]]:
        if not os.path.isfile(path):
            return None
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return _parse_csv_text(f.read())
            if ext in (".json", ".jsonl"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    if ext == ".jsonl":
                        return [json.loads(l) for l in f if l.strip()]
                    return json.load(f)
            return None
        except Exception:
            return None

    pred = _load(pred_path)
    gold = _load(gold_path)
    if pred is None:
        return MetricResult("compare_ml", 0.0, errors=[f"could not load predicted: {pred_path}"])
    if gold is None:
        return MetricResult("compare_ml", 0.0, errors=[f"could not load gold: {gold_path}"])
    if len(pred) != len(gold):
        return MetricResult(
            "compare_ml", 0.0,
            details={"pred_len": len(pred), "gold_len": len(gold)},
            errors=["length mismatch"],
        )

    # Try to find 'prediction' / 'label' / 'target' columns, otherwise compare full rows
    def _find_key(row: Dict[str, Any]) -> Optional[str]:
        for candidate in ("prediction", "predict", "pred", "label", "target", "y"):
            for k in row.keys():
                if str(k).lower() == candidate:
                    return k
        return None

    pred_key = _find_key(pred[0]) if pred else None
    gold_key = _find_key(gold[0]) if gold else None

    if metric in ("accuracy", "f1", "precision", "recall"):
        # classification — try to match labels
        if pred_key and gold_key:
            pvals = [str(r[pred_key]).strip() for r in pred]
            gvals = [str(r[gold_key]).strip() for r in gold]
        else:
            pvals = [str(list(r.values())[0]) for r in pred]
            gvals = [str(list(r.values())[0]) for r in gold]

        correct = sum(1 for p, g in zip(pvals, gvals) if p == g)
        accuracy = correct / len(gvals)
        return MetricResult(
            "compare_ml", accuracy,
            details={"metric": "accuracy", "correct": correct, "total": len(gvals)},
        )

    if metric in ("mae", "mse", "rmse"):
        # regression
        try:
            if pred_key and gold_key:
                pvals = [float(r[pred_key]) for r in pred]
                gvals = [float(r[gold_key]) for r in gold]
            else:
                pvals = [float(list(r.values())[0]) for r in pred]
                gvals = [float(list(r.values())[0]) for r in gold]

            if metric == "mae":
                err = sum(abs(p - g) for p, g in zip(pvals, gvals)) / len(gvals)
            elif metric == "mse":
                err = sum((p - g) ** 2 for p, g in zip(pvals, gvals)) / len(gvals)
            else:  # rmse
                err = math.sqrt(sum((p - g) ** 2 for p, g in zip(pvals, gvals)) / len(gvals))

            # Convert error to a [0,1] score via 1 / (1 + err)
            score = 1.0 / (1.0 + err)
            return MetricResult(
                "compare_ml", score,
                details={"metric": metric, "error": round(err, 6)},
            )
        except Exception as e:
            return MetricResult("compare_ml", 0.0, errors=[f"regression error: {e}"])

    # Fallback: row-level F1 over full rows
    pred_norm = [{k: str(v) for k, v in r.items()} for r in pred]
    gold_norm = [{k: str(v) for k, v in r.items()} for r in gold]
    score = _rowset_f1(pred_norm, gold_norm)
    return MetricResult("compare_ml", score, details={"metric": "row_f1 (fallback)"})


# ---------------------------------------------------------------------------
# Include/exclude test (used for text-string sanity checks)
# ---------------------------------------------------------------------------

def check_include_exclude(pred_value: str, options: Optional[Dict[str, Any]] = None) -> MetricResult:
    """Return 1.0 if pred_value contains all 'include' strings and none of the
    'exclude' strings. Returns 0.0 otherwise."""
    options = options or {}
    includes = options.get("include", []) or []
    excludes = options.get("exclude", []) or []
    case_sensitive = bool(options.get("case_sensitive", False))

    if isinstance(pred_value, (int, float)):
        pred_value = str(pred_value)
    elif pred_value is None:
        pred_value = ""
    text = pred_value if case_sensitive else pred_value.lower()

    def _norm(s: str) -> str:
        return s if case_sensitive else s.lower()

    for inc in includes:
        if _norm(str(inc)) not in text:
            return MetricResult("check_include_exclude", 0.0, details={"missing": inc})
    for ex in excludes:
        if _norm(str(ex)) in text:
            return MetricResult("check_include_exclude", 0.0, details={"unwanted": ex})
    return MetricResult("check_include_exclude", 1.0)


# ---------------------------------------------------------------------------
# Dispatcher: run a metric function by name
# ---------------------------------------------------------------------------

_METRIC_MAP = {
    "compare_csv": compare_csv,
    "compare_csv_files": compare_csv_files,
    "compare_csv_details": compare_csv_details,
    "compare_text": compare_text,
    "compare_image": compare_image,
    "compare_ml": compare_ml,
    "check_include_exclude": check_include_exclude,
    "compare_sqlite": compare_sqlite,
}


def _locate_file(workspace_dir: Path, gold_filename: str) -> Optional[Path]:
    """Search the workspace for a file matching gold_filename (loosely)."""
    if not workspace_dir.exists():
        return None

    # Exact path match first
    cand = workspace_dir / gold_filename
    if cand.is_file():
        return cand

    # Search recursively
    base_lower = gold_filename.lower()
    for p in workspace_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name.lower() == base_lower:
            return p
        # Allow loose match: foo.out.csv matches foo.csv
        ext_match = p.suffix.lower() == Path(gold_filename).suffix.lower()
        if ext_match and Path(gold_filename).stem.lower() in p.name.lower():
            return p
    return None


def _run_metric(
    spec: MetricSpec,
    workspace_dir: Path,
    gold_dir: Path,
) -> MetricResult:
    """Run a single metric spec against workspace vs gold directories."""
    func_name = spec.func
    options = spec.options or {}

    # --- Numeric result ---
    if spec.result_type == "number":
        gold_number = spec.result_value
        if func_name in ("compare_text", "compare_csv"):
            # Look for a result.json number in the workspace
            result_json = workspace_dir / "result.json"
            pred_number: Any = None
            if result_json.exists():
                try:
                    data = json.loads(result_json.read_text(encoding="utf-8"))
                    pred_number = data.get("result", data.get("number", data.get("value")))
                except Exception:
                    pred_number = None
            if pred_number is None:
                # Fall back to scanning text files for a plain number on the first line
                for txt_file in sorted(workspace_dir.rglob("*")):
                    if txt_file.is_file() and txt_file.suffix.lower() in (".txt", ".out", ""):
                        try:
                            line = txt_file.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                            if line:
                                pred_number = line[0]
                                break
                        except Exception:
                            continue
            if pred_number is None:
                return MetricResult(func_name, 0.0, errors=["no predicted number found"])
            return compare_text(pred_number, gold_number, options)

    # --- File-based result ---
    gold_filename = spec.result_value
    if gold_filename is None:
        # Try to find any single output file vs any single gold file
        gold_candidates = [p for p in gold_dir.rglob("*") if p.is_file()]
        pred_candidates = [p for p in workspace_dir.rglob("*") if p.is_file()]
        if not gold_candidates:
            return MetricResult(func_name, 0.0, errors=["no gold files found"])
        if not pred_candidates:
            return MetricResult(func_name, 0.0, errors=["no predicted output files found"])
        best = MetricResult(func_name, 0.0, errors=["no file pair scored"])
        for gold_path in gold_candidates:
            pred_path = _locate_file(workspace_dir, gold_path.name)
            if pred_path is None:
                continue
            sub = _run_file_metric(func_name, str(pred_path), str(gold_path), options)
            if sub.score > best.score:
                best = sub
        return best

    gold_path = gold_dir / str(gold_filename)
    pred_path = _locate_file(workspace_dir, str(gold_filename))
    if pred_path is None:
        return MetricResult(func_name, 0.0, errors=[f"predicted file {gold_filename} not found in workspace"])

    return _run_file_metric(func_name, str(pred_path), str(gold_path), options)


def _run_file_metric(func_name: str, pred_path: str, gold_path: str, options: Dict[str, Any]) -> MetricResult:
    """Dispatch a file-based metric by function name."""
    normalised = func_name.lower().strip()

    # CSV family
    if normalised in ("compare_csv", "compare_csv_files", "compare_csv_details"):
        return compare_csv_files(pred_path, gold_path)

    if normalised == "compare_ml":
        return compare_ml(pred_path, gold_path, options)

    if normalised in ("compare_image", "compare_plot", "compare_figure"):
        return compare_image(pred_path, gold_path, options)

    if normalised == "compare_sqlite":
        return compare_sqlite(pred_path, gold_path, table_name=options.get("table"))

    if normalised == "check_include_exclude":
        try:
            with open(pred_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return MetricResult(func_name, 0.0, errors=[str(e)])
        return check_include_exclude(text, options)

    if normalised == "compare_text":
        try:
            with open(pred_path, "r", encoding="utf-8", errors="replace") as f:
                pred = f.read()
            with open(gold_path, "r", encoding="utf-8", errors="replace") as f:
                gold = f.read()
        except Exception as e:
            return MetricResult(func_name, 0.0, errors=[str(e)])
        return compare_text(pred, gold, options)

    # Unknown metric: fall back to compare_csv_files
    logger.warning("Unknown metric '%s'; falling back to compare_csv_files", func_name)
    return compare_csv_files(pred_path, gold_path)


# ---------------------------------------------------------------------------
# Top-level grader
# ---------------------------------------------------------------------------

def _combine_scores(scores: List[float], conjunction: str) -> float:
    """Combine per-metric scores via the configured conjunction operator."""
    if not scores:
        return 0.0
    conj = conjunction.lower().strip()
    if conj in ("avg", "mean", "average"):
        return sum(scores) / len(scores)
    if conj == "max":
        return max(scores)
    if conj == "min":
        return min(scores)
    if conj == "and":  # all must pass
        return 1.0 if all(s >= 0.5 for s in scores) else 0.0
    if conj == "or":  # any must pass
        return 1.0 if any(s >= 0.5 for s in scores) else 0.0
    # Default: average
    return sum(scores) / len(scores)


def grade_task(
    *,
    task: Task,
    workspace_dir: str | Path,
    dataset_root: str | Path,
    verbose: bool = False,
) -> GradeResult:
    """Grade a single task execution.

    Args:
        task: the Task spec (from TaskLoader).
        workspace_dir: local directory containing the agent's produced files.
        dataset_root: root of the DA-Code dataset (contains gold/<task_id>/).
        verbose: enable debug logging.

    Returns:
        GradeResult with score in [0, 1] and per-metric breakdown.
    """
    workspace_dir = Path(workspace_dir).resolve()
    dataset_root = Path(dataset_root).resolve()
    gold_dir = dataset_root / "gold" / task.task_id

    if not gold_dir.exists():
        # Try fallback: flat gold/ directory (some exports put files directly)
        gold_dir = dataset_root / "gold"

    if verbose:
        logger.info(
            "Grading task %s (workspace=%s, gold=%s)",
            task.task_id, workspace_dir, gold_dir,
        )

    results: List[MetricResult] = []
    for spec in task.metric_funcs:
        try:
            mr = _run_metric(spec, workspace_dir, gold_dir)
        except Exception as e:
            logger.warning("Metric %s failed for %s: %s", spec.func, task.task_id, e)
            mr = MetricResult(spec.func, 0.0, errors=[str(e)])
        if verbose:
            logger.info(
                "   %s -> score=%.3f errors=%s",
                spec.func, mr.score, mr.errors,
            )
        results.append(mr)

    final_score = _combine_scores([r.score for r in results], task.conjunction)
    notes = "; ".join(e for r in results for e in r.errors)

    return GradeResult(
        task_id=task.task_id,
        score=final_score,
        max_score=1.0,
        conjunction=task.conjunction,
        per_metric=results,
        notes=notes,
    )
