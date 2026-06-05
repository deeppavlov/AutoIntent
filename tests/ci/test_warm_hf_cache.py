from __future__ import annotations

import pytest

import warm_hf_cache as wc


class TestParseEntry:
    def test_valid_sha(self):
        repo, rev = wc.parse_entry("prajjwal1/bert-tiny@79779625a0a40f1eee8496e16056bc0d7766df22")
        assert repo == "prajjwal1/bert-tiny"
        assert rev == "79779625a0a40f1eee8496e16056bc0d7766df22"

    def test_missing_sha_raises(self):
        with pytest.raises(wc.ConfigError, match="must be pinned"):
            wc.parse_entry("prajjwal1/bert-tiny")

    def test_non_hex_revision_raises(self):
        with pytest.raises(wc.ConfigError, match="40-char hex"):
            wc.parse_entry("prajjwal1/bert-tiny@main")

    def test_short_revision_raises(self):
        with pytest.raises(wc.ConfigError, match="40-char hex"):
            wc.parse_entry("prajjwal1/bert-tiny@deadbeef")


class TestLoadConfig:
    def test_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("models: []\ndatasets: []\n")
        entries = wc.load_config(cfg)
        assert entries == []

    def test_models_and_datasets(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            "models:\n"
            "  - prajjwal1/bert-tiny@79779625a0a40f1eee8496e16056bc0d7766df22\n"
            "datasets:\n"
            "  - DeepPavlov/clinc150@d835118ecd5ffe5488d22e9e58d1c23d18c33229\n"
        )
        entries = wc.load_config(cfg)
        assert entries == [
            wc.Entry(
                repo_type="model",
                repo_id="prajjwal1/bert-tiny",
                revision="79779625a0a40f1eee8496e16056bc0d7766df22",
            ),
            wc.Entry(
                repo_type="dataset",
                repo_id="DeepPavlov/clinc150",
                revision="d835118ecd5ffe5488d22e9e58d1c23d18c33229",
            ),
        ]

    def test_unknown_top_level_key_raises(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models: []\nfoo: []\n")
        with pytest.raises(wc.ConfigError, match="Unknown top-level"):
            wc.load_config(cfg)

    def test_bad_entry_propagates(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("models:\n  - prajjwal1/bert-tiny\n")
        with pytest.raises(wc.ConfigError, match="must be pinned"):
            wc.load_config(cfg)
