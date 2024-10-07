.DEFAULT_GOAL := all
poetry = poetry run

.PHONY: install
install:
	rm -f poetry.lock
	poetry install --with dev,test,lint

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
	$(poetry) ruff check

.PHONY: all
all: lint