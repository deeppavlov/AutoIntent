.PHONY: all
all: lint

.DEFAULT_GOAL := all

sh = uv run --no-sync --frozen

.PHONY: install
install:
	rm -rf uv.lock 
	uv sync --all-groups

.PHONY: test
test:
	$(sh) pytest tests --cov

.PHONY: test-html
test-html:
	$(sh) pytest --cov --cov-report html

.PHONY: typing
typing:
	$(sh) mypy src/autointent

.PHONY: lint
lint:
	$(sh) ruff format
	$(sh) ruff check --fix

.PHONY: docs
docs:
	$(sh) python -m sphinx build -b html docs/source docs/build/html

.PHONY: test-docs
test-docs:
	$(sh) python -m sphinx build -b doctest docs/source docs/build/html

.PHONY: serve-docs
serve-docs:
	$(sh) python -m http.server -d docs/build/html 8333

.PHONY: multi-version-docs
multi-version-docs:
	$(sh) sphinx-multiversion docs/source docs/build/html

.PHONY: clean-docs
clean-docs:
	rm -rf docs/build
	rm -rf docs/source/autoapi
	rm -rf docs/source/user_guides

.PHONY: schema
schema:
	$(sh) python -m scripts.generate_json_schema_config

