Compute feasibility advisor
===========================

.. note::

   **Experimental.** The advisor's estimates are heuristic and calibrated
   against a limited hardware sample. Treat them as guidance, not guarantees,
   and read :ref:`advisor-accuracy` before relying on a number. The Python
   surface may change in a minor release.

Optimizing a search space can take hours and needs more VRAM than a laptop GPU
has. The advisor answers "will this fit, and how long will it take?" *before*
anything is downloaded or trained.

Command line
------------

Two subcommands. ``inspect`` prices a specific preset or config; ``recommend``
detects your hardware and picks the heaviest bundled preset that still fits.

.. code-block:: bash

   # What will transformers-light cost on this machine?
   autointent-advisor inspect transformers-light

   # ...against a real dataset rather than placeholder sizes
   autointent-advisor inspect transformers-light --dataset banking77

   # Which preset should I use?
   autointent-advisor recommend --dataset banking77

   # Machine-readable output
   autointent-advisor inspect ./my-config.yaml --json

Without ``--dataset``, the advisor uses placeholder dataset sizes
(``--n-samples``, ``--n-classes``, ``--avg-tokens``, ``--task``), so it is
useful before you have assembled any data. ``--budget-vram-gb`` overrides
hardware detection, and ``recommend`` also accepts ``--budget-time-h``.

Both subcommands exit non-zero when nothing is feasible, so they work as a CI
gate.

Reading a report
----------------

Each finding carries a severity:

``ample``
   Comfortably within budget.
``tight``
   Fits, but with little headroom — expect swapping or thermal throttling.
``over``
   Exceeds the budget. Any ``over`` finding makes the whole report infeasible.

The drivers table lists the modules that dominate the cost, so it shows *what*
to change. ``low confidence`` on a report means Hub metadata was unavailable or
incomplete for at least one model, and conservative large-model defaults were
substituted — the numbers are much rougher when you see it.

From Python
-----------

.. code-block:: python

   from autointent import Dataset
   from autointent.advisor import dataset_stats, detect_hardware, estimate, recommend

   report = estimate("transformers-light")
   print(report.is_feasible, report.resource.vram_gb)

   result = recommend(stats=dataset_stats(Dataset.from_json(path)))
   print(result.chosen)

``reduce_to_fit`` goes further: it prunes the most expensive scoring module
repeatedly until the search space fits, raising ``ReduceToFitError`` if nothing
does.

Inside ``Pipeline.fit``
-----------------------

``Pipeline.fit`` accepts a ``preflight`` gate. It defaults to ``"off"``, so the
advisor never runs unless you ask — it makes network calls to the Hugging Face
Hub for model metadata, which does not belong on every fit by default.

.. code-block:: python

   pipeline.fit(dataset, preflight="warn")     # log findings, always continue
   pipeline.fit(dataset, preflight="strict")   # raise PreflightError if infeasible

``"strict"`` raises :class:`autointent.advisor.PreflightError` before allocating
any VRAM, which is the useful mode in CI.

.. _advisor-accuracy:

How accurate is it?
-------------------

Validated end to end on one machine class (RTX 3060 Laptop, 6 GB VRAM / 16 GB
RAM), where all four fitted presets matched their predicted verdict: both
``over`` predictions did run out of memory, and both feasible predictions did
fit. Known limits:

- **Feasibility verdicts are the reliable part.** That is what the advisor was
  built and validated for.
- **VRAM is close but not a guaranteed ceiling.** One preset used 1.22× its
  prediction. Leave headroom rather than trusting the figure exactly.
- **Wall-time estimates are indicative only.** Measured error has run in both
  directions across formula revisions, once by more than an order of magnitude
  for cross-encoders. ``--budget-time-h`` inherits that uncertainty.
- **Preset ranking does not depend on time estimates.** ``recommend`` orders
  presets by a declared cost ranking, so unstable time figures cannot reorder
  its choice.
- **Only one hardware class has been validated end to end.** Treat other
  machines as unverified.
