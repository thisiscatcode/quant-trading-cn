from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterator
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.config import Settings, get_settings
from app.services.files import read_json

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional outside the API image
    psycopg = None
    dict_row = None


MODEL_REGISTRY_SCHEMA_SQL = """
create table if not exists model_versions (
  id text primary key,
  market text not null check (market in ('CN', 'US')),
  model_version text not null,
  profile text not null,
  artifact_path text not null,
  artifact_manifest jsonb not null default '{}'::jsonb,
  trained_at timestamptz not null,
  training_date date not null,
  training_data_start date,
  training_data_end date,
  prediction_as_of date,
  validation_status text not null check (
    validation_status in ('pending', 'passed', 'failed', 'legacy_unreviewed')
  ),
  validation_metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (market, model_version)
);

create index if not exists model_versions_market_profile_idx
  on model_versions(market, profile, trained_at desc);
create index if not exists model_versions_validation_idx
  on model_versions(market, validation_status, trained_at desc);

create table if not exists model_deployments (
  market text primary key check (market in ('CN', 'US')),
  active_model_id text not null references model_versions(id),
  paper_enabled boolean not null default false,
  activated_at timestamptz not null,
  activated_by text not null,
  revision bigint not null default 1,
  updated_at timestamptz not null default now()
);

create table if not exists model_activation_events (
  id text primary key,
  market text not null check (market in ('CN', 'US')),
  previous_model_id text references model_versions(id),
  new_model_id text not null references model_versions(id),
  paper_enabled boolean not null,
  actor text not null,
  reason text,
  revision bigint not null,
  created_at timestamptz not null default now()
);

create index if not exists model_activation_events_market_created_idx
  on model_activation_events(market, created_at desc);
"""

REQUIRED_ARTIFACTS = ("training_metadata.json", "inference_scores_latest.parquet")
OPTIONAL_ARTIFACTS = ("lightgbm_model.txt", "feature_importance.csv")
ACTIVATABLE_VALIDATION_STATUSES = {"passed", "legacy_unreviewed"}


class ModelRegistryError(RuntimeError):
    pass


def normalize_market(value: Any) -> str:
    market = str(value or "").strip().upper()
    if market not in {"CN", "US"}:
        raise ModelRegistryError(f"unsupported model market: {market or 'blank'}")
    return market


def _safe_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return slug or "model"


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_datetime(value: Any, fallback: datetime) -> datetime:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return fallback


@contextmanager
def _connect(*, read_only: bool = False, settings: Settings | None = None) -> Iterator[Any]:
    resolved = settings or get_settings()
    if not resolved.paper_db_url:
        raise ModelRegistryError("PAPER_DB_URL is not configured")
    if psycopg is None or dict_row is None:
        raise ModelRegistryError("psycopg is not installed")
    options = "-c default_transaction_read_only=on" if read_only else None
    with psycopg.connect(
        resolved.paper_db_url,
        row_factory=dict_row,
        connect_timeout=5,
        options=options,
    ) as conn:
        yield conn


def init_model_registry_schema(settings: Settings | None = None) -> None:
    with _connect(settings=settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute(MODEL_REGISTRY_SCHEMA_SQL)
        conn.commit()


def _relative_artifact_path(path: Path, settings: Settings) -> str:
    root = settings.project_root.resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ModelRegistryError(f"model artifact path is outside PROJECT_ROOT: {resolved}") from exc


def resolve_artifact_path(value: Any, settings: Settings | None = None) -> Path:
    resolved_settings = settings or get_settings()
    raw = Path(str(value or "").strip())
    path = raw if raw.is_absolute() else resolved_settings.project_root / raw
    root = resolved_settings.project_root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ModelRegistryError(f"model artifact path escapes PROJECT_ROOT: {resolved}") from exc
    return resolved


def expected_immutable_artifact_path(
    model: dict[str, Any],
    settings: Settings | None = None,
) -> Path:
    resolved = settings or get_settings()
    market = normalize_market(model.get("market"))
    model_version = str(model.get("model_version") or "").strip()
    if not model_version:
        raise ModelRegistryError("model_version is required for immutable artifact resolution")
    return (resolved.quant_dir / "model_registry" / market / model_version).resolve()


def verify_immutable_artifact_location(
    model: dict[str, Any],
    settings: Settings | None = None,
) -> Path:
    resolved = settings or get_settings()
    artifact_dir = resolve_artifact_path(model.get("artifact_path"), resolved)
    expected = expected_immutable_artifact_path(model, resolved)
    if artifact_dir != expected:
        raise ModelRegistryError(
            f"model {model.get('model_version') or 'unknown'} artifact path is mutable; "
            f"expected {expected}"
        )
    return artifact_dir


def _file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"sha256": digest.hexdigest(), "size": stat.st_size}


def build_artifact_manifest(artifact_dir: Path) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for name in (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS):
        path = artifact_dir / name
        if path.exists() and path.is_file():
            manifest[name] = _file_manifest(path)
    missing = [name for name in REQUIRED_ARTIFACTS if name not in manifest]
    if missing:
        raise ModelRegistryError(f"model artifact directory is missing: {', '.join(missing)}")
    return manifest


def manifest_digest(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_artifact_manifest(model: dict[str, Any], settings: Settings | None = None) -> Path:
    artifact_dir = resolve_artifact_path(model.get("artifact_path"), settings)
    expected = model.get("artifact_manifest") if isinstance(model.get("artifact_manifest"), dict) else {}
    current = build_artifact_manifest(artifact_dir)
    for name in expected:
        if current.get(name) != expected.get(name):
            raise ModelRegistryError(f"artifact checksum mismatch for {name}")
    return artifact_dir


def _create_immutable_legacy_snapshot(
    *,
    market: str,
    profile: str,
    source: Path,
    settings: Settings,
) -> dict[str, Any]:
    metadata = read_json(source / "training_metadata.json")
    manifest = build_artifact_manifest(source)
    latest_mtime = max((source / name).stat().st_mtime for name in manifest)
    trained_at = _parse_datetime(metadata.get("trained_at"), datetime.fromtimestamp(latest_mtime, tz=UTC))
    model_version = (
        f"{normalize_market(market).lower()}-{_safe_slug(profile)}-"
        f"{trained_at.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{manifest_digest(manifest)[:8]}"
    )
    destination = expected_immutable_artifact_path(
        {"market": market, "model_version": model_version},
        settings,
    )
    if not destination.exists():
        temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        try:
            for name in manifest:
                shutil.copy2(source / name, temporary / name)
            record = {
                "market": normalize_market(market),
                "model_version": model_version,
                "profile": profile,
                "trained_at": trained_at.isoformat(),
                "validation_status": "legacy_unreviewed",
                "artifact_manifest": build_artifact_manifest(temporary),
            }
            (temporary / "registry_record.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.rename(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return _register_artifact_directory(
        market=market,
        profile=profile,
        artifact_dir=destination,
        validation_status="legacy_unreviewed",
        settings=settings,
        model_version=model_version,
        trained_at=trained_at,
    )


def _model_record_from_artifacts(
    *,
    market: str,
    profile: str,
    artifact_dir: Path,
    validation_status: str,
    model_version: str | None = None,
    trained_at: datetime | None = None,
    settings: Settings,
) -> dict[str, Any]:
    normalized_market = normalize_market(market)
    metadata = read_json(artifact_dir / "training_metadata.json")
    manifest = build_artifact_manifest(artifact_dir)
    latest_mtime = max((artifact_dir / name).stat().st_mtime for name in manifest)
    trained = trained_at or _parse_datetime(metadata.get("trained_at"), datetime.fromtimestamp(latest_mtime, tz=UTC))
    version = model_version or (
        f"{normalized_market.lower()}-{_safe_slug(profile)}-"
        f"{trained.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}-{manifest_digest(manifest)[:8]}"
    )
    model_id = str(uuid5(NAMESPACE_URL, f"aistockcn:{normalized_market}:{version}"))
    training_start = _parse_date(metadata.get("train_date_min"))
    training_end = _parse_date(metadata.get("train_date_max"))
    prediction_as_of = _parse_date(metadata.get("score_date") or metadata.get("inference_date"))
    metrics = metadata.get("metrics") if isinstance(metadata.get("metrics"), dict) else {}
    return {
        "id": model_id,
        "market": normalized_market,
        "model_version": version,
        "profile": profile,
        "artifact_path": _relative_artifact_path(artifact_dir, settings),
        "artifact_manifest": manifest,
        "trained_at": trained,
        "training_date": trained.date(),
        "training_data_start": training_start,
        "training_data_end": training_end,
        "prediction_as_of": prediction_as_of,
        "validation_status": validation_status,
        "validation_metrics": metrics,
    }


def register_model_version(record: dict[str, Any], settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    market = normalize_market(record.get("market"))
    validation_status = str(record.get("validation_status") or "pending")
    if validation_status not in {"pending", "passed", "failed", "legacy_unreviewed"}:
        raise ModelRegistryError(f"invalid validation status: {validation_status}")
    artifact_path = _relative_artifact_path(resolve_artifact_path(record.get("artifact_path"), resolved), resolved)
    with _connect(settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into model_versions (
                  id, market, model_version, profile, artifact_path, artifact_manifest,
                  trained_at, training_date, training_data_start, training_data_end,
                  prediction_as_of, validation_status, validation_metrics
                ) values (
                  %s, %s, %s, %s, %s, %s::jsonb,
                  %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                on conflict (market, model_version) do nothing
                """,
                (
                    str(record["id"]), market, str(record["model_version"]), str(record["profile"]),
                    artifact_path, _json(record.get("artifact_manifest") or {}), record["trained_at"],
                    record["training_date"], record.get("training_data_start"), record.get("training_data_end"),
                    record.get("prediction_as_of"), validation_status, _json(record.get("validation_metrics") or {}),
                ),
            )
            cursor.execute(
                "select * from model_versions where market = %s and model_version = %s",
                (market, str(record["model_version"])),
            )
            row = cursor.fetchone()
        conn.commit()
    if not row:
        raise ModelRegistryError("registered model version could not be loaded")
    if str(row["id"]) != str(record["id"]) or row["artifact_manifest"] != record.get("artifact_manifest"):
        raise ModelRegistryError(f"model version {record['model_version']} already exists with different artifacts")
    return dict(row)


def _register_artifact_directory(
    *,
    market: str,
    profile: str,
    artifact_dir: Path,
    validation_status: str,
    settings: Settings,
    model_version: str | None = None,
    trained_at: datetime | None = None,
) -> dict[str, Any]:
    record = _model_record_from_artifacts(
        market=market,
        profile=profile,
        artifact_dir=artifact_dir,
        validation_status=validation_status,
        model_version=model_version,
        trained_at=trained_at,
        settings=settings,
    )
    return register_model_version(record, settings)


def _registered_model_keys(settings: Settings) -> set[tuple[str, str]]:
    with _connect(read_only=True, settings=settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select market, model_version from model_versions")
            return {(str(row["market"]), str(row["model_version"])) for row in cursor.fetchall()}


def _registry_records(settings: Settings, known: set[tuple[str, str]]) -> list[dict[str, Any]]:
    registered: list[dict[str, Any]] = []
    root = settings.quant_dir / "model_registry"
    for path in sorted(root.glob("*/*/registry_record.json")):
        payload = read_json(path)
        if not payload:
            continue
        market = normalize_market(payload.get("market") or path.parent.parent.name)
        model_version = str(payload.get("model_version") or "").strip()
        if model_version and (market, model_version) in known:
            continue
        artifact_dir = path.parent
        profile = str(payload.get("profile") or read_json(artifact_dir / "training_metadata.json").get("profile_name") or "").strip()
        if not profile:
            continue
        registered.append(
            _register_artifact_directory(
                market=market,
                profile=profile,
                artifact_dir=artifact_dir,
                validation_status=str(payload.get("validation_status") or "pending"),
                settings=settings,
                model_version=model_version or None,
                trained_at=_parse_datetime(payload.get("trained_at"), datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)),
            )
        )
    return registered


def _legacy_profile_records(settings: Settings) -> list[dict[str, Any]]:
    registered: list[dict[str, Any]] = []
    for artifact_dir in sorted((settings.quant_dir / "model_profiles").glob("*/models")):
        if not all((artifact_dir / name).exists() for name in REQUIRED_ARTIFACTS):
            continue
        metadata = read_json(artifact_dir / "training_metadata.json")
        profile = str(metadata.get("profile_name") or artifact_dir.parent.name).strip()
        registered.append(
            _create_immutable_legacy_snapshot(
                market="CN",
                profile=profile,
                source=artifact_dir,
                settings=settings,
            )
        )
    return registered


def _paper_currently_enabled(settings: Settings) -> bool:
    state = read_json(settings.paper_trading_state_path)
    daemon = state.get("daemon") if isinstance(state.get("daemon"), dict) else {}
    return bool(daemon.get("is_running"))


def _ensure_legacy_deployment(registered: list[dict[str, Any]], settings: Settings) -> None:
    if not registered:
        return
    production_dir = settings.models_dir
    if not all((production_dir / name).exists() for name in REQUIRED_ARTIFACTS):
        return
    production_manifest = build_artifact_manifest(production_dir)
    production_metadata = read_json(production_dir / "training_metadata.json")
    active = next(
        (
            row for row in registered
            if str(row.get("profile")) == str(production_metadata.get("profile_name") or "")
            and row.get("artifact_manifest") == production_manifest
        ),
        None,
    )
    if active is None:
        active = _register_artifact_directory(
            market="CN",
            profile=str(production_metadata.get("profile_name") or "legacy_production"),
            artifact_dir=production_dir,
            validation_status="legacy_unreviewed",
            settings=settings,
        )
    paper_enabled = _paper_currently_enabled(settings)
    with _connect(settings=settings) as conn:
        with conn.cursor() as cursor:
            cursor.execute("select active_model_id from model_deployments where market = 'CN' for update")
            if cursor.fetchone() is not None:
                conn.commit()
                return
            cursor.execute(
                """
                insert into model_deployments (
                  market, active_model_id, paper_enabled, activated_at, activated_by, revision
                ) values ('CN', %s, %s, now(), 'legacy_migration', 1)
                """,
                (active["id"], paper_enabled),
            )
            cursor.execute(
                """
                insert into model_activation_events (
                  id, market, previous_model_id, new_model_id, paper_enabled, actor, reason, revision
                ) values (%s, 'CN', null, %s, %s, 'legacy_migration', %s, 1)
                """,
                (
                    str(uuid4()), active["id"], paper_enabled,
                    "Imported the model artifacts actually consumed by production before Model Registry migration.",
                ),
            )
        conn.commit()


def sync_model_registry(settings: Settings | None = None) -> dict[str, Any]:
    resolved = settings or get_settings()
    init_model_registry_schema(resolved)
    deployment = get_active_deployment("CN", settings=resolved, sync=False)
    registered: list[dict[str, Any]] = []
    if deployment is None:
        registered.extend(_legacy_profile_records(resolved))
        _ensure_legacy_deployment(registered, resolved)
    known = _registered_model_keys(resolved)
    registered.extend(_registry_records(resolved, known))
    deployment = get_active_deployment("CN", settings=resolved, sync=False)
    return {"registered": len(registered), "deployment": deployment}


def _deployment_query() -> str:
    return """
      select
        d.market, d.paper_enabled, d.activated_at, d.activated_by, d.revision, d.updated_at,
        v.id as model_id, v.model_version, v.profile, v.artifact_path, v.artifact_manifest,
        v.trained_at, v.training_date, v.training_data_start, v.training_data_end,
        v.prediction_as_of, v.validation_status, v.validation_metrics
      from model_deployments d
      join model_versions v on v.id = d.active_model_id
      where d.market = %s
    """


def get_active_deployment(
    market: str,
    *,
    settings: Settings | None = None,
    sync: bool = True,
) -> dict[str, Any] | None:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    if sync:
        sync_model_registry(resolved)
    with _connect(read_only=True, settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(_deployment_query(), (normalized_market,))
            row = cursor.fetchone()
    return dict(row) if row else None


def list_model_versions(
    market: str,
    *,
    settings: Settings | None = None,
    sync: bool = False,
) -> list[dict[str, Any]]:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    if sync:
        sync_model_registry(resolved)
    with _connect(read_only=True, settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select * from model_versions where market = %s order by trained_at desc, model_version desc",
                (normalized_market,),
            )
            return [dict(row) for row in cursor.fetchall()]


def get_latest_model_for_profile(
    market: str,
    profile: str,
    *,
    settings: Settings | None = None,
    sync: bool = False,
) -> dict[str, Any] | None:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    if sync:
        sync_model_registry(resolved)
    with _connect(read_only=True, settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select * from model_versions
                where market = %s and profile = %s
                order by trained_at desc, model_version desc
                limit 1
                """,
                (normalized_market, str(profile)),
            )
            row = cursor.fetchone()
    return dict(row) if row else None


def resolve_active_model(
    market: str,
    *,
    purpose: str = "picks",
    settings: Settings | None = None,
) -> dict[str, Any]:
    deployment = get_active_deployment(market, settings=settings, sync=False)
    if deployment is None:
        raise ModelRegistryError(f"no active model deployment for {normalize_market(market)}")
    if purpose == "paper" and not deployment.get("paper_enabled"):
        raise ModelRegistryError(f"paper trading is disabled for {deployment['market']} model deployment")
    if purpose == "paper":
        verify_immutable_artifact_location(deployment, settings)
    artifact_dir = verify_artifact_manifest(
        {**deployment, "artifact_manifest": deployment.get("artifact_manifest") or {}},
        settings,
    )
    return {
        **deployment,
        "artifact_dir": str(artifact_dir),
        "scores_path": str(artifact_dir / "inference_scores_latest.parquet"),
        "training_metadata_path": str(artifact_dir / "training_metadata.json"),
    }


def _model_by_reference(cursor: Any, *, market: str, model_version: str | None, profile: str | None) -> dict[str, Any] | None:
    if model_version:
        cursor.execute(
            "select * from model_versions where market = %s and model_version = %s",
            (market, model_version),
        )
    elif profile:
        cursor.execute(
            """
            select * from model_versions
            where market = %s and profile = %s
            order by trained_at desc, model_version desc
            limit 1
            """,
            (market, profile),
        )
    else:
        raise ModelRegistryError("model_version or profile is required")
    row = cursor.fetchone()
    return dict(row) if row else None


def activate_model(
    *,
    market: str,
    model_version: str | None = None,
    profile: str | None = None,
    paper_enabled: bool | None = None,
    actor: str,
    reason: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    sync_model_registry(resolved)
    with _connect(settings=resolved) as conn:
        with conn.cursor() as cursor:
            model = _model_by_reference(
                cursor,
                market=normalized_market,
                model_version=str(model_version or "").strip() or None,
                profile=str(profile or "").strip() or None,
            )
            if model is None:
                raise ModelRegistryError("requested model version was not found")
            if model["validation_status"] not in ACTIVATABLE_VALIDATION_STATUSES:
                raise ModelRegistryError(
                    f"model {model['model_version']} has validation status {model['validation_status']}"
                )
            verify_immutable_artifact_location(model, resolved)
            verify_artifact_manifest(model, resolved)
            cursor.execute(
                "select active_model_id, paper_enabled, revision from model_deployments where market = %s for update",
                (normalized_market,),
            )
            previous = cursor.fetchone()
            previous_model_id = previous["active_model_id"] if previous else None
            effective_paper = bool(previous["paper_enabled"]) if previous and paper_enabled is None else bool(paper_enabled)
            revision = int(previous["revision"] if previous else 0) + 1
            cursor.execute(
                """
                insert into model_deployments (
                  market, active_model_id, paper_enabled, activated_at, activated_by, revision, updated_at
                ) values (%s, %s, %s, now(), %s, %s, now())
                on conflict (market) do update set
                  active_model_id = excluded.active_model_id,
                  paper_enabled = excluded.paper_enabled,
                  activated_at = excluded.activated_at,
                  activated_by = excluded.activated_by,
                  revision = excluded.revision,
                  updated_at = now()
                """,
                (normalized_market, model["id"], effective_paper, actor, revision),
            )
            cursor.execute(
                """
                insert into model_activation_events (
                  id, market, previous_model_id, new_model_id, paper_enabled, actor, reason, revision
                ) values (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid4()), normalized_market, previous_model_id, model["id"], effective_paper,
                    actor, reason, revision,
                ),
            )
        conn.commit()
    deployment = get_active_deployment(normalized_market, settings=resolved, sync=False)
    if deployment is None:
        raise ModelRegistryError("model activation committed without a deployment row")
    return deployment


def update_validation_status(
    *,
    market: str,
    model_version: str,
    validation_status: str,
    metrics: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    status = str(validation_status or "").strip()
    if status not in {"pending", "passed", "failed", "legacy_unreviewed"}:
        raise ModelRegistryError(f"invalid validation status: {status}")
    with _connect(settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                update model_versions
                set validation_status = %s,
                    validation_metrics = case when %s::jsonb = '{}'::jsonb then validation_metrics else %s::jsonb end
                where market = %s and model_version = %s
                returning *
                """,
                (status, _json(metrics or {}), _json(metrics or {}), normalized_market, model_version),
            )
            row = cursor.fetchone()
        conn.commit()
    if row is None:
        raise ModelRegistryError("requested model version was not found")
    return dict(row)


def list_activation_events(
    market: str,
    *,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    with _connect(read_only=True, settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select e.*, previous.model_version as previous_model_version,
                       current.model_version as new_model_version
                from model_activation_events e
                left join model_versions previous on previous.id = e.previous_model_id
                join model_versions current on current.id = e.new_model_id
                where e.market = %s
                order by e.created_at desc
                limit %s
                """,
                (normalized_market, max(min(int(limit), 200), 1)),
            )
            return [dict(row) for row in cursor.fetchall()]


def rollback_model(
    *,
    market: str,
    actor: str,
    reason: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    resolved = settings or get_settings()
    normalized_market = normalize_market(market)
    with _connect(read_only=True, settings=resolved) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select previous.model_version, d.paper_enabled
                from model_deployments d
                join model_activation_events e
                  on e.market = d.market and e.revision = d.revision
                join model_versions previous on previous.id = e.previous_model_id
                where d.market = %s
                """,
                (normalized_market,),
            )
            row = cursor.fetchone()
    if row is None:
        raise ModelRegistryError(f"no previous {normalized_market} model is available for rollback")
    return activate_model(
        market=normalized_market,
        model_version=str(row["model_version"]),
        paper_enabled=bool(row["paper_enabled"]),
        actor=actor,
        reason=reason,
        settings=resolved,
    )
