from __future__ import annotations

import pytest
import warm_hf_cache as wc


class TestResolveEntry:
    def test_known_repo_resolves_to_pinned_sha(self):
        entry = wc._resolve_entry("prajjwal1/bert-tiny", "model")
        assert entry.repo_id == "prajjwal1/bert-tiny"
        assert entry.repo_type == "model"
        # The SHA must match DEFAULT_REVISIONS — we look it up here rather
        # than hardcoding to keep this test honest if the pin moves.
        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS
        assert entry.revision == DEFAULT_REVISIONS["prajjwal1/bert-tiny"]

    def test_unknown_repo_raises(self):
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._resolve_entry("not-a-real/repo", "model")


class TestLoadConfig:
    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("models: []\ndatasets: []\n")
        entries = wc._load_config(cfg)
        assert entries == []

    def test_models_resolved_via_default_revisions(self, tmp_path):
        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models:\n"
            "  - prajjwal1/bert-tiny\n"
            "datasets: []\n"
        )
        entries = wc._load_config(cfg)
        assert entries == [
            wc.Entry(
                repo_type="model",
                repo_id="prajjwal1/bert-tiny",
                revision=DEFAULT_REVISIONS["prajjwal1/bert-tiny"],
            ),
        ]

    def test_unknown_top_level_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models: []\nfoo: []\n")
        with pytest.raises(wc.ConfigError, match="Unknown top-level"):
            wc._load_config(cfg)

    def test_unpinned_model_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models:\n  - not-a-real/repo\n")
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._load_config(cfg)

    def test_dataset_entry_raises(self, tmp_path):
        # DEFAULT_REVISIONS covers models only today. If we ever need to
        # warm a dataset, _resolve_entry must be extended; the parser
        # raises until then so a dataset in the YAML can't silently
        # regress to an unpinned download.
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models: []\n"
            "datasets:\n"
            "  - DeepPavlov/clinc150\n"
        )
        with pytest.raises(wc.ConfigError, match="not in DEFAULT_REVISIONS"):
            wc._load_config(cfg)


class TestPrewarmConfigsAreSubsetOfDefaultRevisions:
    """Defense-in-depth: the warm-cache job itself raises ConfigError on
    unknown repo IDs (via _resolve_entry), but that error only fires at
    CI time. This test catches the same drift at unit-test time so a
    misconfigured YAML never reaches the warm-cache job."""

    @pytest.mark.parametrize("yaml_path", [".ci/hf-prewarm.yaml"])
    def test_every_model_is_pinned_in_default_revisions(self, yaml_path):
        from pathlib import Path

        import yaml as pyyaml

        from autointent.configs._pinned_revisions import DEFAULT_REVISIONS

        repo_root = Path(__file__).resolve().parents[2]
        data = pyyaml.safe_load((repo_root / yaml_path).read_text()) or {}
        models = data.get("models") or []
        missing = [m for m in models if m not in DEFAULT_REVISIONS]
        assert not missing, (
            f"{yaml_path}: {missing} not in DEFAULT_REVISIONS. Add a pin to "
            f"src/autointent/configs/_pinned_revisions.py."
        )

    @pytest.mark.parametrize("yaml_path", [".ci/hf-prewarm.yaml"])
    def test_no_sha_suffix_in_repo_ids(self, yaml_path):
        """The new YAML format is bare repo IDs. A '@' in an entry means
        someone added an entry in the old 'repo@sha' format — likely
        because they copy-pasted from git history. Catch it explicitly so
        the error message points at the right fix."""
        from pathlib import Path

        import yaml as pyyaml

        repo_root = Path(__file__).resolve().parents[2]
        data = pyyaml.safe_load((repo_root / yaml_path).read_text()) or {}
        with_sha = [m for m in (data.get("models") or []) if "@" in m]
        assert not with_sha, (
            f"{yaml_path}: {with_sha} use the old 'repo@sha' format. "
            "Drop the '@<sha>' suffix; SHAs are looked up from "
            "DEFAULT_REVISIONS by warm_hf_cache.py."
        )
