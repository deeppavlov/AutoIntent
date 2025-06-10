# Contribute to AutoIntent

## Minimum Configuration

We use `poetry` as our dependency manager and packager.

1. Install `poetry`. We recommend referring to the official documentation section [Installation with the official installer](https://python-poetry.org/docs/#installing-with-the-official-installer). In short, you just need to run:
```bash
curl -sSL https://install.python-poetry.org | python3 -
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
poetry run pytest tests/modules/scoring/test_bert.py
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
make update
```

## Building Documentation

Build the HTML version in the `docs/build` folder:
```bash
make docs
```

Build the HTML version and host it locally:
```bash
make serve-docs
```
