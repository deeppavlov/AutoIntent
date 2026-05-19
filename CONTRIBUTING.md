# Contribute to AutoIntent

## Minimum Configuration

We use `uv` as our dependency manager and packager.

1. Install `uv`. We recommend referring to the documentations on [installing uv](https://docs.astral.sh/uv/#installation). In short, you just need to run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the project and navigate to the root directory

3. Install the project with all dependencies:
```bash
make install
```

## Additional Setup

To make it easier to track code style errors, we recommend installing the ruff extension for your IDE. For example, for VSCode:
```
https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff
```
With this extension, code style errors will be underlined directly in the editor.

A `.vscode/settings.json` file has been added to the project root, which points the extension to the linter configuration.

## Contribute

1. Create a branch for your work. To make it easier for others to understand the nature of your contribution, use brief but clear names. We recommend starting branch names with `feat/` for new features, `fix/` for bug fixes, `refactor/` for refactoring, and `test/` for adding tests.

2. Commit, commit, commit, commit

3. If there are new features, it's advisable to add tests for them in the [tests](./tests) directory.

4. You can open a PR!

Every commit in any PR triggers github actions with automated tests. All checks block merging into the main branch (with rare exceptions).

Sometimes waiting for CI can be long, and sometimes it's more convenient to run individual tests:
- Check that your changes don't break existing features
```bash
make test
```
Or run a specific test (using `test_bert.py` as an example):
```bash
uv run pytest tests/modules/scoring/test_bert.py
```
- Check code style (it also applies formatter)
```bash
make lint
```
- Check type hints:
```bash
make typing
```
Note: If mypy shows different errors locally compared to github actions, you should update your local dependencies:
```bash
make install
```
But it still doesn't guarantee that the local type checker will give the same errors as CI. This is because CI is configured to check on Python 3.10 and your local python version is probably the latest one.

## Building Documentation

Build the HTML version in the `docs/build` folder:
```bash
make docs
```

Build the HTML version and host it locally:
```bash
make serve-docs
```

## Preparing documentation for a release

Use this checklist when cutting a new package release (for example `0.3.0`). Documentation is published to [deeppavlov.github.io/AutoIntent](https://deeppavlov.github.io/AutoIntent/) via GitHub Pages; a **published GitHub Release** triggers the multi-version build and deploy.

### Before opening a PR

1. **Align versions** across `pyproject.toml`, `docs/source/conf.py` (`release`), and `CHANGELOG.md`.
2. **Update prose** under `docs/source/` (`.rst` files).
3. **Update tutorials** in the repo-root `user_guides/` directory — not under `docs/source/user_guides/`, which is generated at build time and gitignored.
4. **Run doctests** (same as CI on PRs and pushes to `dev`):
   ```bash
   make test-docs
   ```
5. **Build HTML locally** and fix any errors:
   ```bash
   make docs
   ```
   Optional preview:
   ```bash
   make serve-docs
   ```
   If the build is stale or autoapi / tutorial links look wrong:
   ```bash
   make clean-docs
   make docs
   ```
6. **Match CI dependencies** when tutorials or API pages fail on missing imports:
   ```bash
   uv sync --group docs --extra catboost --extra peft --extra transformers --extra sentence-transformers --extra openai
   ```
   **Pandoc** is required for nbsphinx (CI installs it via `apt`).
7. **Regenerate the optimizer JSON schema** if `OptimizerConfig` or related Pydantic models changed:
   ```bash
   make schema
   ```

### Do not commit

- `docs/build/`
- `docs/source/autoapi/`
- `docs/source/user_guides/` (symlinks and notebook run artifacts)
- `**/__pycache__/`

### Version switcher (`versions.json`)

`docs/_static/versions.json` is **auto-generated** on every Sphinx build from git tags matching `vX.Y.Z` (see `docs/source/docs_utils/versions_generator.py`). Do not hand-edit it for a release. Until the `vX.Y.Z` tag exists, local builds will still list the previous tag as stable — that is expected.

### Release day

1. Merge documentation and code changes into **`dev`** (CI runs `make test-docs` and `make docs`).
2. Create a git tag **`vX.Y.Z`** on the release commit (must match `v` + semver, for example `v0.3.0`).
3. **Publish a GitHub Release** for that tag. This triggers:
   - PyPI publish (`.github/workflows/release.yaml`)
   - Multi-version docs build and deploy (`.github/workflows/build-docs.yaml` → `make multi-version-docs` → GitHub Pages under `/versions/`).
4. Verify the live site: the version switcher shows the new release as **stable**, and `https://deeppavlov.github.io/AutoIntent/versions/vX.Y.Z/` loads.

To dry-run the multi-version build locally (requires full git history and tags):

```bash
make multi-version-docs
```
