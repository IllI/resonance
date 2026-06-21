#!/usr/bin/env python3
"""Build provenance inventories for the TRC TPU research journal."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_journal" / "generated"


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=text, encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def write_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def commits_and_files(branch: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    fmt = "%H%x1f%P%x1f%aI%x1f%an%x1f%s"
    lines = git("log", branch, "--reverse", f"--format={fmt}").splitlines()
    commits: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    for sequence, line in enumerate(lines, 1):
        sha, parents, date, author, subject = line.split("\x1f", 4)
        commits.append({
            "sequence": sequence, "commit": sha, "parents": parents,
            "date": date, "author": author, "subject": subject,
        })
        raw = git("diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M", sha)
        for item in raw.splitlines():
            parts = item.split("\t")
            status = parts[0]
            if status.startswith("R") and len(parts) >= 3:
                old_path, path = parts[1], parts[2]
            else:
                old_path, path = "", parts[-1]
            changes.append({
                "sequence": sequence, "commit": sha, "date": date,
                "status": status, "path": path, "old_path": old_path,
            })
    return commits, changes


def markdown_manifest(branch: str, changes: list[dict[str, object]]) -> list[dict[str, object]]:
    paths = [p for p in git("ls-tree", "-r", "--name-only", branch).splitlines()
             if Path(p).suffix.lower() in {".md", ".markdown"}]
    history: dict[str, list[dict[str, object]]] = {}
    for row in changes:
        history.setdefault(str(row["path"]), []).append(row)
    rows: list[dict[str, object]] = []
    for path in paths:
        content = git("show", f"{branch}:{path}")
        heading = next((line.lstrip("# ").strip() for line in content.splitlines()
                        if line.startswith("#")), "")
        events = history.get(path, [])
        rows.append({
            "path": path,
            "bytes": len(content.encode("utf-8")),
            "first_commit": events[0]["commit"] if events else "",
            "first_date": events[0]["date"] if events else "",
            "last_commit": events[-1]["commit"] if events else "",
            "last_date": events[-1]["date"] if events else "",
            "heading": heading,
        })
    return rows


def classify(path: str) -> str:
    p = path.lower()
    if p.endswith((".md", ".markdown")):
        return "documentation"
    if "/results" in p or "_results/" in p or p.endswith((".json", ".csv", ".npz", ".npy")):
        return "result_or_data"
    if p.endswith((".py", ".ps1", ".sh")):
        return "code_or_launcher"
    if p.endswith((".yaml", ".yml", ".toml")):
        return "configuration"
    if p.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        return "figure_or_reference"
    if p.endswith((".log", ".txt")):
        return "log_or_text"
    return "other"


def working_tree_manifest() -> list[dict[str, object]]:
    raw = git("status", "--porcelain=v1", "-z", "--untracked-files=all", text=False)
    entries = raw.decode("utf-8", "replace").split("\0")
    rows: list[dict[str, object]] = []
    for entry in entries:
        if not entry:
            continue
        status, path = entry[:2], entry[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == "research_journal" or path.startswith("research_journal/"):
            continue
        full = ROOT / path
        try:
            stat = full.stat()
            size, mtime = stat.st_size, int(stat.st_mtime)
        except OSError:
            size, mtime = "", ""
        rows.append({
            "status": status, "path": path.replace(os.sep, "/"),
            "kind": classify(path.replace(os.sep, "/")),
            "bytes": size, "mtime_epoch": mtime,
        })
    return sorted(rows, key=lambda row: str(row["path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="quantum-teleportation-results")
    args = parser.parse_args()
    commits, changes = commits_and_files(args.branch)
    docs = markdown_manifest(args.branch, changes)
    live = working_tree_manifest()
    write_csv("commits.csv", ["sequence", "commit", "parents", "date", "author", "subject"], commits)
    write_csv("commit_files.csv", ["sequence", "commit", "date", "status", "path", "old_path"], changes)
    write_csv("markdown_documents.csv", ["path", "bytes", "first_commit", "first_date", "last_commit", "last_date", "heading"], docs)
    write_csv("working_tree_artifacts.csv", ["status", "path", "kind", "bytes", "mtime_epoch"], live)
    print(f"Wrote {len(commits)} commits, {len(changes)} file changes, {len(docs)} docs, {len(live)} live artifacts to {OUT}")


if __name__ == "__main__":
    main()
