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

.PHONY: docs
docs:
	cd docs && make html

.PHONY: serve-docs
serve-docs:
	cd docs/_build/html && $(poetry) python -m http.server

.PHONY: all
all: lint