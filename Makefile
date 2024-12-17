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

.PHONY: sync
sync:
	poetry install --sync --with dev,test,lint,typing,docs

.PHONY: docs
docs:
	$(poetry) python -m sphinx build -b html docs/source docs/build/html

.PHONY: test-docs
test-docs: docs
	$(poetry) python -m sphinx build -b doctest docs/source docs/build/html

.PHONY: serve-docs
serve-docs: docs
	$(poetry) python -m http.server -d docs/build/html 8333

.PHONY: multi-version
multi-version:
	$(poetry) python -m sphinx-multiversion docs/source docs/build/html

.PHONY: clean-docs
clean-docs:
	rm -rf docs/build
	rm -rf docs/source/autoapi
	rm -rf docs/source/user_guides

.PHONY: all
all: lint