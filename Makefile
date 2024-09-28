.DEFAULT_GOAL := all
poetry = poetry run

.PHONY: test
test:
	$(poetry) pytest

.PHONY: lint
lint:
	$(poetry) ruff format
	$(poetry) ruff check

.PHONY: all
all: lint