.DEFAULT_GOAL := all
poetry = poetry run

.PHONY: install
install:
	poetry install --with dev,test,lint,typing,docs

.PHONY: test
test:
	$(poetry) pytest tests --cov

.PHONY: test-html
test-html:
	$(poetry) pytest --cov --cov-report html

.PHONY: typing
typing:
	$(poetry) mypy autointent

.PHONY: lint
lint:
	$(poetry) ruff format
	$(poetry) ruff check --fix

.PHONY: all
all: lint

.PHONY: sync
sync:
	poetry install --sync --with dev,test,lint,typing,docs

.PHONY: docs-serve
docs-serve:
	$(poetry) sphinx-autobuild docs/source docs/build/html