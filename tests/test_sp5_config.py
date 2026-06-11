"""SP-5 config reads the shared Postgres DSN and the MinIO settings from the environment."""
import importlib


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import htr_sp5.config as cfg
    return importlib.reload(cfg)


def test_pg_dsn_defaults_and_table_names(monkeypatch):
    cfg = _reload(monkeypatch, HTR_PG_DSN="postgresql://u:p@h:5432/db")
    assert cfg.PG_DSN == "postgresql://u:p@h:5432/db"
    assert cfg.EVAL_RUN_TABLE == "eval_run"
    assert cfg.EVAL_RESULT_TABLE == "eval_result"
    assert cfg.UPLOAD_TABLE == "upload_result"


def test_minio_settings_from_env(monkeypatch):
    cfg = _reload(
        monkeypatch,
        HTR_MINIO_ENDPOINT="localhost:9000",
        HTR_MINIO_ACCESS_KEY="ak",
        HTR_MINIO_SECRET_KEY="sk",
        HTR_MINIO_BUCKET="htr-uploads",
        HTR_MINIO_SECURE="false",
    )
    assert cfg.MINIO_ENDPOINT == "localhost:9000"
    assert cfg.MINIO_ACCESS_KEY == "ak"
    assert cfg.MINIO_SECRET_KEY == "sk"
    assert cfg.MINIO_BUCKET == "htr-uploads"
    assert cfg.MINIO_SECURE is False


def test_minio_configured_flag(monkeypatch):
    cfg = _reload(monkeypatch, HTR_MINIO_ENDPOINT="", HTR_MINIO_ACCESS_KEY="", HTR_MINIO_SECRET_KEY="")
    assert cfg.minio_configured() is False


def test_minio_configured_flag_true(monkeypatch):
    # All three required fields present → helper must return True.
    cfg = _reload(monkeypatch,
                  HTR_MINIO_ENDPOINT="localhost:9000",
                  HTR_MINIO_ACCESS_KEY="ak",
                  HTR_MINIO_SECRET_KEY="sk")
    assert cfg.minio_configured() is True


def test_minio_secure_true_is_case_and_whitespace_insensitive(monkeypatch):
    # "true" (and variants like " TRUE ") must parse to the boolean True,
    # mirroring the lowercase "false" → False behaviour tested above.
    cfg = _reload(monkeypatch, HTR_MINIO_SECURE=" TRUE ")
    assert cfg.MINIO_SECURE is True
