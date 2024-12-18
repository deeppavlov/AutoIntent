import json
import logging
import os
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sorted_versions(repo_root: Path, base_url: str) -> list[dict]:
    os.chdir(repo_root)
    # Fetch tags and branches
    tags_output = subprocess.check_output(["git", "tag"], text=True).strip().split("\n")  # noqa: S603, S607
    tag_regex = re.compile(r"^v\d+\.\d+\.\d+$")  # Matches tags like v0.0.1
    tags = [tag for tag in tags_output if tag_regex.match(tag)]

    sorted_tags = sorted(tags, reverse=True)

    logger.info("sorted_tags: %s", sorted_tags)

    # Prepare versions list (similar to previous implementation)
    versions = []

    if sorted_tags:
        versions.append(
            {
                "name": f"{sorted_tags[0]} (stable)",
                "version": sorted_tags[0],
                "url": f"{base_url}/{sorted_tags[0]}/",
                "preferred": True,
            }
        )

    for tag in sorted_tags[1:]:
        versions.append({"version": tag, "url": f"{base_url}/{tag}/"})  # noqa: PERF401

    # Get branches
    branches_output = subprocess.check_output(["git", "branch", "-r"], text=True).strip().split("\n")  # noqa: S603, S607
    dev_branches = [
        branch.strip().split("/")[-1] for branch in branches_output if "origin/dev" in branch and "HEAD" not in branch
    ]
    logger.info("dev_branches: %s", dev_branches)

    for branch in dev_branches:
        versions.append({"version": f"{branch} (dev)", "url": f"{base_url}/{branch}/"})  # noqa: PERF401

    return versions


def generate_versions_json(repo_root: Path, base_url: str) -> None:
    """
    Sphinx extension to generate versions.json during documentation build.
    """
    # Only generate if no exception occurred
    # Ensure _static directory exists
    static_dir = repo_root / "docs" / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # Generate and write versions JSON
    versions = get_sorted_versions(repo_root, base_url)
    versions_path = static_dir / "versions.json"

    # Always generate a JSON, even if empty
    with versions_path.open("w") as f:
        json.dump(versions, f, indent=4)

    logger.info("versions.json generated at %s", versions_path)
