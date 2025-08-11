sh = uv run
groups = --group test --group typing --group lint --group nb --group docs

.PHONY: install
install:
	rm -rf uv.lock 
	uv lock
	uv sync $(groups)

.PHONY: test
test:
	$(uv) pytest tests --cov

.PHONY: test-html
test-html:
	$(uv) pytest --cov --cov-report html

.PHONY: typing
typing:
	$(uv) mypy autointent

.PHONY: lint
lint:
	$(uv) ruff format
	$(uv) ruff check --fix

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

