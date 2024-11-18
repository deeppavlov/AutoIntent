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
	$(poetry) sphinx-apidoc -e -E -f --remove-old -o docs/source/apiref autointent
	$(poetry) python -m sphinx build docs/source docs/build/html

.PHONY: serve-docs
serve-docs: docs
	$(poetry) python -m http.server -d docs/build/html 8333