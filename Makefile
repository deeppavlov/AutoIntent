.DEFAULT_GOAL := all
poetry = poetry run

.PHONY: test
test:
	$(poetry) pytest tests --cov

.PHONY: test-html
test-html:
	$(poetry) pytest --cov-report html

.PHONY: typing
typing:
	$(poetry) mypy autointent

.PHONY: lint
lint:
	$(poetry) ruff format
	$(poetry) ruff check

.PHONY: all
all: lint