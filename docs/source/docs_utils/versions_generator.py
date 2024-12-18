import os
import json
import subprocess
from typing import List, Dict

import re
import os
import json
import subprocess


def get_sorted_versions(repo_root, base_url):
    os.chdir(repo_root)
    # Fetch tags and branches
    tags_output = subprocess.check_output(['git', 'tag'], text=True).strip().split('\n')
    tag_regex = re.compile(r"^v\d+\.\d+\.\d+$")  # Matches tags like v0.0.1
    tags = [tag for tag in tags_output if tag_regex.match(tag)]

    sorted_tags = sorted(tags, reverse=True)

    # Prepare versions list (similar to previous implementation)
    versions = []

    if sorted_tags:
        versions.append({
            "name": f"{sorted_tags[0]} (stable)",
            "version": sorted_tags[0],
            "url": f"{base_url}/{sorted_tags[0]}/",
            "preferred": True,
        })

    for tag in sorted_tags[1:]:
        versions.append({
            "version": tag,
            "url": f"{base_url}/{tag}/"
        })

    # Get branches
    branches_output = subprocess.check_output(['git', 'branch', '-r'], text=True).strip().split('\n')
    dev_branches = [
        branch.strip().split('/')[-1]
        for branch in branches_output
        if 'origin/dev' in branch and "HEAD" not in branch
    ]
    print(dev_branches, branches_output)

    for branch in dev_branches:
        versions.append({
            "version": f"{branch} (dev)",
            "url": f"{base_url}/{branch}/"
        })

    return versions


def generate_versions_json(app, repo_root, base_url):
    """
    Sphinx extension to generate versions.json during documentation build.
    """
    # Only generate if no exception occurred
    # Ensure _static directory exists
    print(app.srcdir)
    static_dir = os.path.join(repo_root, "docs", '_static')
    os.makedirs(static_dir, exist_ok=True)

    # Generate and write versions JSON
    versions = get_sorted_versions(repo_root, base_url)
    versions_path = os.path.join(static_dir, 'versions.json')

    # Always generate a JSON, even if empty
    with open(versions_path, 'w') as f:
        json.dump(versions, f, indent=4)

    print(f"Generated versions.json with {len(versions)} versions")
