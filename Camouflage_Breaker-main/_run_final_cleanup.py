"""
Build pre-delete manifest and execute approved TUKU cleanup.
Does NOT touch protected production assets.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CB = Path(r"C:\Users\batka\Downloads\TUKU DEEP LEARING\Camouflage_Breaker-main")
TUKU_ROOT = Path(r"C:\Users\batka\Downloads\TUKU DEEP LEARING")
DOWNLOADS = Path(r"C:\Users\batka\Downloads")

PROTECTED = [
    CB / "saved_models" / "resunet_best.pth",
    CB / "saved_models" / "sinetv2" / "sinetv2_cod10k_best.pth",
    CB / "saved_models" / "classifier_best.pth",
    CB / "saved_models" / "class_mapping.json",
]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path, category: str, reason: str, dep: str) -> dict:
    size = path.stat().st_size if path.exists() else 0
    return {
        "absolute_path": str(path),
        "relative_path": str(path.relative_to(TUKU_ROOT)) if str(path).startswith(str(TUKU_ROOT)) else str(path),
        "file_size": size,
        "sha256": sha256_file(path) if path.is_file() else None,
        "file_type": "file" if path.is_file() else ("directory" if path.is_dir() else "missing"),
        "reason": reason,
        "dependency_search_result": dep,
        "classification": category,
    }


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def main():
    planned: list[dict] = []
    retained_uncertain: list[dict] = []

    # --- UNCERTAIN / KEEP (document only) ---
    retained_uncertain.extend(
        [
            {
                "path": r"C:\Users\batka\Downloads\archive (5).zip",
                "reason": "SHA differs from archive (6).zip despite same size — not byte-identical duplicate",
                "sha256": "8d79f8c30e0a0bba97c17f261b93c590c377541025094218a93b93ddbfa05f2d",
            },
            {
                "path": r"C:\Users\batka\Downloads\archive (6).zip",
                "reason": "SHA differs from archive (5).zip — keep both",
                "sha256": "3a7674e09d44951dab7051bf041cc46146bfabae2db3d9ad803ce1fafc4cdbbc",
            },
            {
                "path": "/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_final.pth",
                "reason": "SHA differs from approved best (7ed2e38d… vs d6ffec2d…) — unique weights; referenced by finetune script",
                "sha256": "7ed2e38dfdc376def7ced4081c1915c3dbcce0a829d989e1315716c400d9eb4f",
            },
            {
                "path": str(CB / "saved_models" / "backup_pre_continue_archive6_20260920"),
                "reason": "Historical SHA conflict evidence — KEEP",
            },
            {
                "path": "/home/batka/ESCNet/checkpoints/escnet/epoch_120.pth",
                "reason": "Official ESCNet baseline — UNCERTAIN/KEEP",
            },
        ]
    )

    # --- Individual files to delete ---
    file_deletes = [
        (
            CB / "saved_models" / "resunet_epoch_5.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Intermediate ResUNet epoch; production best exists",
            "No hardcoded path refs in app/pipeline; only SAVE_INTERVAL writer",
        ),
        (
            CB / "saved_models" / "resunet_epoch_10.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Intermediate ResUNet epoch; production best exists",
            "No hardcoded path refs in app/pipeline",
        ),
        (
            CB / "saved_models" / "resunet_epoch_15.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Intermediate ResUNet epoch; production best exists",
            "No hardcoded path refs in app/pipeline",
        ),
        (
            CB / "saved_models" / "sinetv2" / "sinetv2_cod10k_best_smoke_backup.pth",
            "TEMPORARY_ARTIFACT",
            "SINet smoke backup; SHA != production",
            "Only mentioned in cleanup audit docs",
        ),
        (
            CB
            / "outputs"
            / "archive6_resunet_manifest_s42"
            / "checkpoints"
            / "resunet_archive6_epoch_5.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Archive-6 intermediate; best retained",
            "Written by train script; best+eval kept",
        ),
        (
            CB
            / "outputs"
            / "archive6_resunet_manifest_s42"
            / "checkpoints"
            / "resunet_archive6_epoch_10.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Archive-6 intermediate; best retained",
            "Written by train script; best+eval kept",
        ),
        (
            CB
            / "outputs"
            / "archive6_resunet_manifest_s42"
            / "checkpoints"
            / "resunet_archive6_epoch_15.pth",
            "INTERMEDIATE_CHECKPOINT",
            "Archive-6 intermediate; best retained",
            "Written by train script; best+eval kept",
        ),
        (
            CB / "models" / "sinetv2" / "source" / "AWESOME_COD_LIST.md",
            "TEMPORARY_ARTIFACT",
            "Upstream paper list; unused by production",
            "Docs only; not imported",
        ),
    ]

    for path, cat, reason, dep in file_deletes:
        if path.exists():
            planned.append(file_entry(path, cat, reason, dep))

    # log files (temporary consoles)
    log_temps = [
        CB / "logs" / "resunet_train.log",
        CB / "logs" / "sinetv2_train.log",
        CB / "logs" / "classifier_train.log",
        CB / "logs" / "final_eval.log",
        CB / "logs" / "generate_crops_train.log",
        CB / "logs" / "remaining_pipeline.log",
        CB / "logs" / "watch_and_continue.log",
        CB / "outputs" / "archive6_resunet_official_test_eval_run.log",
        CB / "outputs" / "archive6_resunet_official_test_eval" / "console.log",
        CB / "outputs" / "archive6_resunet_manifest_s42" / "logs" / "train_console.log",
        CB / "outputs" / "continue_archive6" / "resunet.log",
        CB / "outputs" / "continue_archive6" / "sinetv2.log",
        CB / "outputs" / "continue_archive6" / "classifier.log",
    ]
    for path in log_temps:
        if path.exists():
            planned.append(
                file_entry(
                    path,
                    "TEMPORARY_ARTIFACT",
                    "Temporary/debug console log; summaries retained in JSON/MD",
                    "Not imported by production code",
                )
            )

    # directories
    dir_deletes = [
        (
            CB / "models" / "sinetv2" / "source" / "jittor_lib",
            "TEMPORARY_ARTIFACT",
            "Jittor port unused by production PyTorch wrapper",
            "Not imported by sinetv2_wrapper/pipeline/app",
        ),
        (
            CB / "models" / "sinetv2" / "source" / "imgs",
            "TEMPORARY_ARTIFACT",
            "Upstream marketing images",
            "Not imported",
        ),
        (
            CB / "outputs" / "sinetv2",
            "TEMPORARY_ARTIFACT",
            "SINet smoke-test output images",
            "Only test_sinetv2_inference writes here",
        ),
        (
            TUKU_ROOT / ";",
            "TEMPORARY_ARTIFACT",
            "Empty accidental directory",
            "None",
        ),
        (
            TUKU_ROOT / "cp",
            "TEMPORARY_ARTIFACT",
            "Empty accidental directory",
            "None",
        ),
        (
            DOWNLOADS / "archive",
            "DUPLICATE_DATASET",
            "Near-empty extract stub",
            "None",
        ),
        (
            DOWNLOADS / "archive (1)",
            "DUPLICATE_DATASET",
            "Near-empty extract stub",
            "None",
        ),
        (
            DOWNLOADS / "archive (2)",
            "DUPLICATE_DATASET",
            "Empty extract stub",
            "None",
        ),
        (
            DOWNLOADS / "TUKU_COD10K_ARCHIVE_SOURCE",
            "DUPLICATE_DATASET",
            "Extracted COD10K used for completed Archive-6 experiment; production dataset/ retained; best+eval retained",
            "Only referenced by completed Archive-6 run_config/prepare_status/train script paths",
        ),
    ]

    for path, cat, reason, dep in dir_deletes:
        if path.exists():
            planned.append(
                {
                    "absolute_path": str(path),
                    "relative_path": str(path),
                    "file_size": dir_size(path),
                    "sha256": None,
                    "file_type": "directory",
                    "reason": reason,
                    "dependency_search_result": dep,
                    "classification": cat,
                    "note": "Directory — per-file SHA omitted; size is recursive sum",
                }
            )

    # __pycache__ dirs
    for pyc in CB.rglob("__pycache__"):
        if pyc.is_dir():
            # skip if under protected external annotation? still safe
            planned.append(
                {
                    "absolute_path": str(pyc),
                    "relative_path": str(pyc.relative_to(CB)),
                    "file_size": dir_size(pyc),
                    "sha256": None,
                    "file_type": "directory",
                    "reason": "Python bytecode cache",
                    "dependency_search_result": "None",
                    "classification": "SAFE_TO_DELETE",
                }
            )
    for pyc in CB.rglob("*.pyc"):
        if pyc.is_file():
            planned.append(file_entry(pyc, "SAFE_TO_DELETE", "Bytecode", "None"))

    # WSL deletions (recorded as remote paths)
    wsl_dirs = [
        (
            "/home/batka/C3Net",
            "REJECTED_MODEL_INFRASTRUCTURE",
            "Rejected C3Net source+checkpoints; no TUKU production imports",
            "No active Windows/WSL ESCNet refs",
        ),
        (
            "/home/batka/tuku_c3net",
            "REJECTED_MODEL_INFRASTRUCTURE",
            "Rejected C3Net conda env",
            "No production refs",
        ),
        (
            "/home/batka/ESCNet/checkpoints/escnet_smoke",
            "TEMPORARY_ARTIFACT",
            "ESCNet smoke checkpoint dir",
            "Only _escnet_step10_smoke.py (historical smoke)",
        ),
        (
            "/home/batka/tmp",
            "TEMPORARY_ARTIFACT",
            "WSL temporary directory",
            "None",
        ),
    ]

    for wsl_path, cat, reason, dep in wsl_dirs:
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc", f"du -sb {wsl_path} 2>/dev/null | cut -f1"],
            capture_output=True,
            text=True,
        )
        size = int(r.stdout.strip() or "0")
        planned.append(
            {
                "absolute_path": wsl_path,
                "relative_path": wsl_path,
                "file_size": size,
                "sha256": None,
                "file_type": "directory_wsl",
                "reason": reason,
                "dependency_search_result": dep,
                "classification": cat,
            }
        )

    # Assert protected still present and record
    protected_snap = {}
    for p in PROTECTED:
        assert p.is_file(), f"PROTECTED MISSING: {p}"
        protected_snap[str(p)] = {"sha256": sha256_file(p), "size": p.stat().st_size}

    # ESCNet best via WSL
    r = subprocess.run(
        [
            "wsl",
            "-e",
            "bash",
            "-lc",
            "sha256sum /home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth; stat -c%s /home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth",
        ],
        capture_output=True,
        text=True,
    )
    lines = [x for x in r.stdout.strip().splitlines() if x]
    esc_sha = lines[0].split()[0]
    esc_size = int(lines[1])
    protected_snap["/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth"] = {
        "sha256": esc_sha,
        "size": esc_size,
    }
    assert esc_sha == "d6ffec2d4d281846bd55016dbed6c8a5a938ba0a126158bef5f9949c92f55e82"

    # Archive-6 best must remain
    a6 = (
        CB
        / "outputs"
        / "archive6_resunet_manifest_s42"
        / "checkpoints"
        / "resunet_archive6_manifest_best.pth"
    )
    assert a6.is_file(), "Archive-6 best missing"
    assert (CB / "outputs" / "archive6_resunet_official_test_eval" / "summary.json").is_file()

    total_bytes = sum(int(x.get("file_size") or 0) for x in planned)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "PREDELETE_MANIFEST",
        "planned_deletion_count": len(planned),
        "planned_reclaim_bytes": total_bytes,
        "protected_snapshot": protected_snap,
        "retained_uncertain": retained_uncertain,
        "items": planned,
        "rules": "No UNCERTAIN items included. Production/COD10K/TUKU_v2/EXTERNAL_TEST_24/backup_pre_continue/escnet_final/zips kept.",
    }

    man_path = CB / "PROJECT_CLEANUP_PREDELETE_MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST {man_path} items={len(planned)} bytes={total_bytes}")

    # Update protected before file with ESCNet
    before_path = CB / "PROJECT_CLEANUP_PROTECTED_HASHES_BEFORE.json"
    before_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "files": protected_snap,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    deleted_rows = []
    errors = []

    def delete_path(path: Path, meta: dict):
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                errors.append({"path": str(path), "error": "missing_at_delete_time"})
                return
            deleted_rows.append(
                {
                    "path": str(path),
                    "size_bytes": meta.get("file_size") or 0,
                    "sha256": meta.get("sha256") or "",
                    "category": meta.get("classification"),
                    "reason": meta.get("reason"),
                    "dependency_check": meta.get("dependency_search_result"),
                    "deletion_status": "DELETED",
                }
            )
            print(f"DELETED {path}")
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            print(f"FAIL {path}: {exc}")

    # Execute Windows deletions from planned (skip WSL)
    for item in planned:
        p = Path(item["absolute_path"])
        if item.get("file_type") == "directory_wsl":
            continue
        # never touch protected
        if any(p.resolve() == prot.resolve() for prot in PROTECTED if prot.exists()):
            errors.append({"path": str(p), "error": "REFUSED_PROTECTED"})
            continue
        if "external_24" in str(p).lower() or "external_test_24" in str(p).lower():
            errors.append({"path": str(p), "error": "REFUSED_EXTERNAL_TEST"})
            continue
        if "TUKU_DATASET_V2" in str(p):
            errors.append({"path": str(p), "error": "REFUSED_TUKU_V2"})
            continue
        if str(p).replace("\\", "/").endswith("/dataset") or "\\dataset\\" in str(p):
            # only refuse the production dataset folder itself
            if CB / "dataset" in p.parents or p == CB / "dataset":
                errors.append({"path": str(p), "error": "REFUSED_DATASET"})
                continue
        delete_path(p, item)

    # WSL deletions
    for item in planned:
        if item.get("file_type") != "directory_wsl":
            continue
        wsl_path = item["absolute_path"]
        # never delete ESCNet root or finetune best
        if "escnet_finetune" in wsl_path and "smoke" not in wsl_path:
            errors.append({"path": wsl_path, "error": "REFUSED_ESCNET_FINETUNE"})
            continue
        if wsl_path.rstrip("/") == "/home/batka/ESCNet":
            errors.append({"path": wsl_path, "error": "REFUSED_ESCNET_ROOT"})
            continue
        if wsl_path.rstrip("/") == "/home/batka/tuku_escnet":
            errors.append({"path": wsl_path, "error": "REFUSED_TUKU_ESCNET_ENV"})
            continue
        cmd = f"rm -rf -- '{wsl_path}'"
        r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
        if r.returncode == 0:
            deleted_rows.append(
                {
                    "path": wsl_path,
                    "size_bytes": item.get("file_size") or 0,
                    "sha256": "",
                    "category": item.get("classification"),
                    "reason": item.get("reason"),
                    "dependency_check": item.get("dependency_search_result"),
                    "deletion_status": "DELETED",
                }
            )
            print(f"DELETED_WSL {wsl_path}")
        else:
            errors.append({"path": wsl_path, "error": r.stderr or r.stdout or str(r.returncode)})
            print(f"FAIL_WSL {wsl_path}: {r.stderr}")

    # Post hashes
    after = {}
    for p in PROTECTED:
        after[str(p)] = {"sha256": sha256_file(p), "size": p.stat().st_size if p.is_file() else None, "exists": p.is_file()}
    r = subprocess.run(
        [
            "wsl",
            "-e",
            "bash",
            "-lc",
            "test -f /home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth && sha256sum /home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth",
        ],
        capture_output=True,
        text=True,
    )
    after_esc = r.stdout.split()[0] if r.returncode == 0 and r.stdout.strip() else None
    after["/home/batka/ESCNet/checkpoints/escnet_finetune/escnet_cod10k_best.pth"] = {
        "sha256": after_esc,
        "exists": after_esc is not None,
    }

    unchanged = True
    for k, v in protected_snap.items():
        av = after.get(k, {})
        if v.get("sha256") != av.get("sha256"):
            unchanged = False
            print(f"HASH_MISMATCH {k}")

    result = {
        "deleted_count": len(deleted_rows),
        "reclaimed_bytes": sum(int(r["size_bytes"]) for r in deleted_rows),
        "errors": errors,
        "protected_unchanged": unchanged,
        "protected_after": after,
        "deleted_rows": deleted_rows,
        "retained_uncertain": retained_uncertain,
    }
    (CB / "PROJECT_CLEANUP_DELETE_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("deleted_count", "reclaimed_bytes", "protected_unchanged", "errors")}, indent=2))
    return 0 if unchanged and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
