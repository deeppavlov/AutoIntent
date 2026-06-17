Inference servers
=================

AutoIntent can serve a **trained** pipeline behind two optional interfaces:

* **HTTP (FastAPI)** — a small REST API for ``predict`` and health checks. Use this when you integrate with services, gateways, or clients that speak HTTP/JSON.
* **MCP (FastMCP)** — a Model Context Protocol server with tools (``predict``, ``classes``, ``train_data``). Use this when an LLM host or IDE connects over MCP (stdio for local tools, or HTTP transport for remote access).

Both servers load assets from a directory on disk (the same folder produced when you optimize and dump a pipeline). They are **not** a replacement for training: you must fit or load a pipeline and write it to that directory first.

Installation
------------

Install the core package, then add the extra that matches the server you need:

.. code-block:: bash

   pip install "autointent[fastapi]"

.. code-block:: bash

   pip install "autointent[fastmcp]"

The ``fastapi`` extra pulls in FastAPI, Uvicorn, and ``pydantic-settings``. The ``fastmcp`` extra pulls in FastMCP and ``pydantic-settings``.

.. note::

   If you use ``uv``, the project declares **incompatible optional extras** for ``codecarbon`` and ``fastmcp`` (see ``tool.uv.conflicts`` in ``pyproject.toml``). You cannot enable both in the same resolved environment; pick one or use separate virtual environments.

Prerequisites
-------------

* A directory containing a **saved optimized pipeline** (for example the project directory after ``context.dump()`` from optimization, or another path where ``Pipeline.load`` succeeds).
* For the **MCP** server only: a ``dataset.json`` file inside that same directory (the server loads training metadata and samples for the ``classes`` and ``train_data`` tools).

Configuration (both servers)
----------------------------

Settings are defined with ``pydantic-settings`` and the prefix ``AUTOINTENT_``. Values can be set in the process environment or in a ``.env`` file in the current working directory.

**Shared**

``AUTOINTENT_PATH`` *(required)* — filesystem path to the pipeline directory (same meaning as the ``path`` field in code).

**HTTP server**

``AUTOINTENT_HOST`` — bind address (default ``127.0.0.1``).

``AUTOINTENT_PORT`` — listen port (default ``8013``).

**MCP server**

``AUTOINTENT_TRANSPORT`` — ``stdio`` (default) or ``http``.

``AUTOINTENT_HOST`` / ``AUTOINTENT_PORT`` — used when ``AUTOINTENT_TRANSPORT=http`` (defaults ``127.0.0.1`` and ``8012``).

Example ``.env``:

.. code-block:: text

   AUTOINTENT_PATH=/path/to/my_autointent_project
   # Optional HTTP defaults:
   # AUTOINTENT_HOST=0.0.0.0
   # AUTOINTENT_PORT=8013
   # Optional MCP over HTTP:
   # AUTOINTENT_TRANSPORT=http
   # AUTOINTENT_PORT=8012

Set these variables **before** starting the process (the HTTP app reads settings at import time).

HTTP server (FastAPI)
---------------------

**Run with Uvicorn** (recommended; module path matches the FastAPI instance):

.. code-block:: bash

   uvicorn autointent.server.http:app --host 127.0.0.1 --port 8013

Bind address and port can follow your deployment; ensure ``AUTOINTENT_PATH`` still points at the pipeline directory.

**Run via the module entrypoint** (uses ``AUTOINTENT_HOST`` and ``AUTOINTENT_PORT`` from settings):

.. code-block:: bash

   python -c "from autointent.server.http import main; main()"

Endpoints
.........

* ``GET /health`` — returns ``{"status": "healthy"}``.
* ``POST /predict`` — JSON body and response shaped like the Pydantic models below.

**Request** (``PredictRequest``): ``{"utterances": ["text one", "text two"]}``

**Response** (``PredictResponse``): ``{"predictions": [...]}`` — one prediction per input utterance.

Predictions follow the same convention as ``Pipeline.predict``:

* Single-label: each item is an integer class id, or ``null`` for out-of-scope.
* Multi-label: each item is a list of integer class ids, or ``null`` for out-of-scope.

MCP server (FastMCP)
--------------------

**Stdio (default)** — typical for MCP clients that spawn a subprocess:

.. code-block:: bash

   python -c "from autointent.server.mcp import main; main()"

With ``AUTOINTENT_TRANSPORT`` unset or ``stdio``, ``main()`` calls ``mcp.run()`` with stdio transport.

**HTTP transport** — set ``AUTOINTENT_TRANSPORT=http`` (and optionally host/port). ``main()`` then runs with ``transport="http"`` so clients can connect to the configured TCP port (default ``8012``).

Tools
.....

* ``predict`` — arguments: ``utterances: list[str]``. Returns ``predictions`` in the same sense as the HTTP API.
* ``classes`` — pagination: ``page``, ``page_size``. Returns ``classes`` (list of ``Intent`` objects: ``id``, ``name``, ``tags``, regex fields, ``description``) and ``pagination_info``.
* ``train_data`` — pagination and optional ``class_filter`` (list of class ids). Returns ``samples`` (``id``, ``text``, ``label``) and ``pagination_info``.

See the :doc:`API reference <autoapi/autointent/index>` for full type details.
