#!/usr/bin/env python3
"""Validate a local Hugging Face ONNX package (offline; no network, no upload, no auth).

Checks structure, metadata, checksums, ONNX validity, ONNX Runtime execution + dynamic axes,
standalone inference, and the absence of leaked absolute paths / secrets / forbidden files.
Self-contained: depends only on stdlib + numpy + onnx + onnxruntime (not the bert_cord package).

Usage:
  python scripts/validate_hf_onnx_package.py bert-cord-27m-mlm-onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

REQUIRED_FILES = ["README.md", "LICENSE", "config.json", "evaluation.json", "requirements.txt",
                  "inference.py", "MANIFEST.json", "onnx/model.onnx", "onnx/model.onnx.data"]
EXPECTED_INPUTS = {"input_ids", "attention_mask", "token_type_ids"}
# Patterns that must NOT appear as files inside the package.
FORBIDDEN_PATTERNS = [r"\.pt$", r"\.pth$", r"\.ckpt$", r"\.safetensors$", r"optimizer",
                      r"scheduler", r"rng", r"\.env$", r"\.key$", r"\.pem$", r"wandb",
                      r"experiments", r"\.venv", r"__pycache__", r"\.git(/|$)"]
# Absolute-path leaks + secret-ish markers to scan text/JSON for.
ABS_PATH_RE = re.compile(r"(/Users/|/home/[^\s\"]+|/sessions/|[A-Za-z]:\\\\)")
SECRET_RES = [re.compile(r"wandb_api_key", re.I), re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
              re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
              re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
              re.compile(r"\b(api[_-]?key|secret|password)\b\s*[:=]\s*[\"'][^\"']{6,}", re.I)]
TEXT_EXTS = {".md", ".json", ".txt", ".py"}


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def validate(pkg: str) -> Report:
    import numpy as np
    import onnx
    import onnxruntime as ort

    r = Report()
    pkg = os.path.abspath(pkg)

    # 1. Required files exist.
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(pkg, f))]
    r.add("required files present", not missing, f"missing: {missing}" if missing else "")

    # 2. Forbidden files absent.
    found_forbidden = []
    for root, _dirs, names in os.walk(pkg):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), pkg).replace(os.sep, "/")
            if any(re.search(p, rel) for p in FORBIDDEN_PATTERNS):
                found_forbidden.append(rel)
    r.add("no forbidden files", not found_forbidden,
          f"found: {found_forbidden}" if found_forbidden else "")

    # 3. JSON parses.
    cfg = _load_json(pkg, "config.json", r, "config.json parses")
    ev = _load_json(pkg, "evaluation.json", r, "evaluation.json parses")
    man = _load_json(pkg, "MANIFEST.json", r, "MANIFEST.json parses")

    # 4. README front matter.
    readme = _read(os.path.join(pkg, "README.md"))
    r.add("README YAML front matter", readme.lstrip().startswith("---") and
          "library_name: onnxruntime" in readme)

    # 5. Packaging-source commit matches current git HEAD (when in a git repo). Model-source
    #    provenance is tracked separately and is NOT required to equal HEAD.
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head and cfg:
        r.add("packaging source commit matches HEAD",
              cfg.get("packaging_source_commit") == head,
              f"config={cfg.get('packaging_source_commit')} head={head}")
    else:
        r.add("packaging source commit matches HEAD", True, "skipped (not a git repo)")

    # 6. Checksums match MANIFEST.
    if man and "files" in man:
        bad = []
        for entry in man["files"]:
            fp = os.path.join(pkg, entry["path"])
            if not os.path.exists(fp) or _sha256(fp) != entry["sha256"] or \
                    os.path.getsize(fp) != entry["size_bytes"]:
                bad.append(entry["path"])
        # MANIFEST must NOT list itself.
        listed = {e["path"] for e in man["files"]}
        r.add("MANIFEST checksums match", not bad and "MANIFEST.json" not in listed,
              f"mismatched: {bad}" if bad else
              ("MANIFEST lists itself" if "MANIFEST.json" in listed else ""))
    else:
        r.add("MANIFEST checksums match", False, "no files list")

    # 7. Both ONNX files exist + external linkage resolves.
    graph = os.path.join(pkg, "onnx", "model.onnx")
    data = os.path.join(pkg, "onnx", "model.onnx.data")
    r.add("both ONNX files exist", os.path.exists(graph) and os.path.exists(data))
    try:
        m = onnx.load(graph, load_external_data=False)
        locs = {e.value for t in m.graph.initializer for e in t.external_data
                if e.key == "location"}
        r.add("external data linkage -> model.onnx.data", locs == {"model.onnx.data"},
              f"locations={locs}")
    except Exception as e:  # noqa: BLE001
        r.add("external data linkage -> model.onnx.data", False, str(e))

    # 8. ONNX checker.
    try:
        onnx.checker.check_model(onnx.load(graph, load_external_data=True))
        r.add("onnx.checker passes", True)
    except Exception as e:  # noqa: BLE001
        r.add("onnx.checker passes", False, str(e))

    # 9. ORT load + contract + dynamic axes.
    try:
        sess = ort.InferenceSession(graph, providers=["CPUExecutionProvider"])
        names = {i.name for i in sess.get_inputs()}
        outs = [o.name for o in sess.get_outputs()]
        r.add("ORT loads + input/output contract",
              EXPECTED_INPUTS.issubset(names) and "logits" in outs,
              f"inputs={sorted(names)} outputs={outs}")

        def run(b, s):
            # Small safe id range valid for any vocab >= 8 (values are irrelevant here).
            ii = np.random.randint(0, 8, (b, s)).astype(np.int64)
            am = np.ones_like(ii)
            tt = np.zeros_like(ii)
            out = sess.run(["logits"], {"input_ids": ii, "attention_mask": am,
                                        "token_type_ids": tt})[0]
            return out.shape

        s1 = run(1, 16)
        s2 = run(3, 16)  # dynamic batch
        s3 = run(1, 40)  # dynamic sequence
        r.add("dynamic batch inference", s1[0] == 1 and s2[0] == 3, f"{s1} vs {s2}")
        r.add("dynamic sequence inference", s1[1] == 16 and s3[1] == 40, f"{s1} vs {s3}")
    except Exception as e:  # noqa: BLE001
        r.add("ORT loads + input/output contract", False, str(e))

    # 10. inference.py runs in a subprocess.
    try:
        proc = subprocess.run([sys.executable, "inference.py"], cwd=pkg,
                              capture_output=True, text=True, timeout=120)
        r.add("inference.py runs (subprocess)", proc.returncode == 0,
              "" if proc.returncode == 0 else proc.stderr.strip().splitlines()[-1:] and
              proc.stderr.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        r.add("inference.py runs (subprocess)", False, str(e))

    # 11. No leaked absolute paths / secrets in text + JSON files.
    leaks, secrets = [], []
    for root, _dirs, names in os.walk(pkg):
        for n in names:
            if os.path.splitext(n)[1].lower() not in TEXT_EXTS:
                continue
            txt = _read(os.path.join(root, n))
            rel = os.path.relpath(os.path.join(root, n), pkg)
            if ABS_PATH_RE.search(txt):
                leaks.append(rel)
            if any(rx.search(txt) for rx in SECRET_RES):
                secrets.append(rel)
    r.add("no private absolute paths", not leaks, f"in: {leaks}" if leaks else "")
    r.add("no apparent secrets/keys", not secrets, f"in: {secrets}" if secrets else "")

    # 12. Report package size.
    total = sum(os.path.getsize(os.path.join(root, n))
                for root, _d, names in os.walk(pkg) for n in names)
    print(f"[info] package total size: {total:,} bytes ({total/1024/1024:.1f} MB)")
    return r


def _load_json(pkg, name, r: Report, label: str):
    try:
        with open(os.path.join(pkg, name), encoding="utf-8") as fh:
            obj = json.load(fh)
        r.add(label, True)
        return obj
    except Exception as e:  # noqa: BLE001
        r.add(label, False, str(e))
        return None


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    p = argparse.ArgumentParser(description="Validate a local HF ONNX package (offline).")
    p.add_argument("package", help="Path to the package directory.")
    args = p.parse_args()
    if not os.path.isdir(args.package):
        print(f"[validate_hf] not a directory: {args.package}", file=sys.stderr)
        return 2
    print("=" * 68)
    print(f"[validate_hf] validating {args.package} (offline; no network)")
    print("-" * 68)
    rep = validate(args.package)
    print("-" * 68)
    n_ok = sum(1 for _, ok, _ in rep.checks if ok)
    print(f"[validate_hf] {n_ok}/{len(rep.checks)} checks passed -> "
          f"{'PASS' if rep.ok else 'FAIL'}")
    print("=" * 68)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
