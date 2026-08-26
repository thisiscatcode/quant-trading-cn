#!/usr/bin/env python3
"""Reconcile latest model picks with a Futu gateway paper-trading agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

try:
    import psycopg
    from psycopg.types.json import Jsonb
except Exception:  # pragma: no cover - optional runtime dependency for DB persistence
    psycopg = None
    Jsonb = None

from control_settings import (
    exclude_st_from_model_candidates,
    filter_model_candidate_rows,
    is_st_stock_name,
)
from execution_model import (
    DEFAULT_EXECUTION_MODEL,
    ExecutionModel,
    buy_liquidity_skip_reason,
    execution_model_snapshot,
    execution_model_with_limit_bps,
    liquidity_cap_notional,
    marketable_limit_price,
    near_price_limit,
    previous_close_from_row,
)
from trading_fees import DEFAULT_FEE_MODEL, transaction_fee


DEFAULT_SCORES_PATH = "quant_data/models/inference_scores_latest.parquet"
DEFAULT_STATE_DIR = "quant_data/paper_trading"
DAILY_SNAPSHOTS_FILENAME = "daily_snapshots.json"
DEFAULT_MARKET = "CN"
DEFAULT_GATEWAY_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_AGENT_ID = "aistockcn-paper-cn"
DEFAULT_AGENT_KEY = "local-dev-agent-key"
DEFAULT_AGENT_ID_HEADER = "X-Agent-Id"
DEFAULT_AGENT_KEY_HEADER = "X-Agent-Key"
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.5
DEFAULT_LOT_SIZE = 100
DEFAULT_CASH_BUFFER_PCT = 0.05
DEFAULT_MAX_ORDER_QTY = 1000
DEFAULT_STRATEGY_ID = "aistock_rebalance"
QUANT_ORDER_REMARK_PREFIX = "aistock sig="
ORDER_ATTEMPT_INTERVAL_SECONDS = 2.2
SINA_QUOTE_URL = "https://hq.sinajs.cn/list={code}"
SINA_QUOTE_REFERER = "https://finance.sina.com.cn"
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
ACTIVE_TRADING_WINDOWS = (
    (dt_time(9, 35, 0), dt_time(11, 30, 0)),
    (dt_time(13, 0, 0), dt_time(14, 55, 0)),
)

TERMINAL_ORDER_STATUSES = {
    "CANCELLED",
    "CANCELLED_ALL",
    "CANCELLED_PART",
    "CANCELLED_PART_ALL",
    "DELETED",
    "DISABLED",
    "EXPIRED",
    "FAILED",
    "FILLED_ALL",
    "REJECTED",
    "SUBMIT_FAILED",
}

PRICE_LIMIT_ERROR_TEXTS = (
    "\u62a5\u5355\u4ef7\u683c\u4e0d\u5728\u6da8\u8dcc\u505c\u533a\u95f4",
    "not in the limit move",
    "price is not in the limit",
)
ORDER_RATE_LIMIT_ERROR_TEXTS = (
    "high frequency",
    "maximum 15 times per 30 seconds",
)

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_beijing() -> datetime:
    return datetime.now(BEIJING_TZ)


def parse_trade_datetime(value: Any) -> datetime | None:
    if value in (None, "", "NaT"):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        timestamp = pd.to_datetime(text, errors="coerce")
        if pd.isna(timestamp):
            return None
        parsed = timestamp.to_pydatetime()
    if parsed.tzinfo is not None:
        return parsed.astimezone(BEIJING_TZ)
    if "T" in text:
        return parsed.replace(tzinfo=timezone.utc).astimezone(BEIJING_TZ)
    return parsed.replace(tzinfo=BEIJING_TZ)


def trade_date_text(value: Any = None) -> str:
    parsed = parse_trade_datetime(value)
    if parsed is None:
        parsed = now_beijing()
    return str(parsed.date())


def is_active_trading_hours(moment: datetime | None = None) -> bool:
    current = moment or now_beijing()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    beijing = current.astimezone(BEIJING_TZ)
    if beijing.weekday() >= 5:
        return False
    beijing_time = beijing.time()
    return any(start <= beijing_time <= end for start, end in ACTIVE_TRADING_WINDOWS)


def normalize_date_text(value: Any) -> str | None:
    if value in (None, "", "NaT"):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return str(pd.Timestamp(parsed).date())


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def normalize_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        return text.split(".", 1)[-1]
    return text.zfill(6) if text.isdigit() else text


def normalize_status(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    for old, new in [("-", "_"), (" ", "_"), ("/", "_")]:
        text = text.replace(old, new)
    return text


def is_active_order(status: Any) -> bool:
    normalized = normalize_status(status)
    return bool(normalized) and normalized not in TERMINAL_ORDER_STATUSES


def to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "N/A"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def round_price(value: float) -> float:
    return round(float(value), 2)


def clamp_non_negative(value: float) -> float:
    return max(float(value), 0.0)


def build_reference_price(base_price: float) -> float:
    if base_price <= 0:
        return 0.0
    return round_price(base_price)


def build_marketable_limit_price(
    latest_price: float,
    side: str,
    execution_model: ExecutionModel = DEFAULT_EXECUTION_MODEL,
) -> float:
    return marketable_limit_price(latest_price, side, execution_model)


def is_price_limit_error(message: str) -> bool:
    normalized = str(message or "").lower()
    return any(text.lower() in normalized for text in PRICE_LIMIT_ERROR_TEXTS)


def is_order_rate_limit_error(message: str) -> bool:
    normalized = str(message or "").lower()
    return any(text.lower() in normalized for text in ORDER_RATE_LIMIT_ERROR_TEXTS)


def sina_exchange_prefix(symbol: str, exchange: Any = None) -> str:
    exchange_text = str(exchange or "").strip().upper()
    if exchange_text.startswith(("SH", "SSE")):
        return "sh"
    if exchange_text.startswith(("SZ", "SZE")):
        return "sz"
    normalized = normalize_symbol(symbol)
    if normalized.startswith(("5", "6", "9")):
        return "sh"
    return "sz"


def sina_quote_code(symbol: str, exchange: Any = None) -> str:
    return f"{sina_exchange_prefix(symbol, exchange)}{normalize_symbol(symbol)}"


def parse_sina_quote_price(payload: str, quote_code: str) -> float:
    text = str(payload or "").strip()
    prefix = f"var hq_str_{quote_code}="
    if not text.startswith(prefix):
        raise GatewayError(f"Sina quote response did not contain {quote_code}: {text[:120]}")
    try:
        quote_text = text.split('"', 2)[1]
    except IndexError as exc:
        raise GatewayError(f"Sina quote response was malformed for {quote_code}: {text[:120]}") from exc
    fields = quote_text.split(",")
    if len(fields) < 4 or not fields[0]:
        raise GatewayError(f"Sina quote response was empty for {quote_code}: {text[:120]}")
    latest_price = to_float(fields[3])
    if latest_price <= 0:
        raise GatewayError(f"Sina quote latest price unavailable for {quote_code}: {quote_text[:120]}")
    return latest_price


def get_sina_latest_price(symbol: str, exchange: Any = None) -> float:
    quote_code = sina_quote_code(symbol, exchange)
    request = Request(
        SINA_QUOTE_URL.format(code=quote_code),
        headers={
            "Referer": SINA_QUOTE_REFERER,
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urlopen(request, timeout=5) as response:
            body = response.read().decode("gb18030", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GatewayError(f"Sina quote request failed for {quote_code}: HTTP {exc.code} {detail}") from exc
    except URLError as exc:
        raise GatewayError(f"Sina quote request failed for {quote_code}: {exc.reason}") from exc
    return parse_sina_quote_price(body, quote_code)


def choose_balance_record(records: list[dict[str, Any]], account_id: int | None) -> dict[str, Any]:
    if not records:
        return {}
    if account_id is None:
        return records[0]
    for record in records:
        if str(record.get("acc_id") or record.get("account_id") or "") == str(account_id):
            return record
    return records[0]


def extract_balance_metrics(records: list[dict[str, Any]], account_id: int | None) -> dict[str, float | str | None]:
    record = choose_balance_record(records, account_id)
    power_keys = ["power", "buying_power", "max_power_short", "available_funds", "avl_withdrawal_cash"]
    cash_keys = ["cash", "cash_balance", "cash_and_cash_equivalents", "available_cash", "withdraw_cash"]
    asset_keys = ["total_assets", "total_asset", "assets", "net_assets", "market_val"]

    def first(keys: list[str]) -> float:
        for key in keys:
            if key in record:
                value = to_float(record.get(key), default=float("nan"))
                if not math.isnan(value):
                    return value
        return 0.0

    return {
        "power": first(power_keys),
        "cash": first(cash_keys),
        "total_assets": first(asset_keys),
        "currency": str(record.get("currency") or record.get("base_currency") or DEFAULT_MARKET),
    }


def score_file_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def quant_dir_for_scores_path(path: Path) -> Path:
    resolved = Path(path)
    if resolved.parent.name == "models" and resolved.parent.parent.name != "model_profiles":
        return resolved.parent.parent
    if len(resolved.parents) >= 4 and resolved.parent.name == "models" and resolved.parent.parent.parent.name == "model_profiles":
        return resolved.parent.parent.parent.parent
    return resolved.parent.parent


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 1)


def _active_training_metadata_path(scores_path: Path) -> Path:
    return Path(scores_path).parent / "training_metadata.json"


def _model_profile_catalog_path() -> Path:
    return Path("run") / "model_profiles.json"


def active_rebalance_profile(scores_path: Path) -> dict[str, Any]:
    metadata = read_json(_active_training_metadata_path(scores_path))
    profile_name = str(metadata.get("profile_name") or "").strip()
    label_horizon = _positive_int(metadata.get("label_horizon"), 1)
    catalog = read_json(_model_profile_catalog_path())
    profiles = catalog.get("profiles") if isinstance(catalog.get("profiles"), list) else []

    if not profile_name:
        profile_name = str(catalog.get("default_profile") or "").strip()

    for profile in profiles:
        if isinstance(profile, dict) and str(profile.get("name") or "").strip() == profile_name:
            return {
                "profile_name": profile_name,
                "profile_label": str(profile.get("label") or profile_name),
                "rebalance_every": _positive_int(profile.get("backtest_rebalance_every"), label_horizon),
                "label_horizon": _positive_int(profile.get("label_horizon"), label_horizon),
            }

    return {
        "profile_name": profile_name or "unknown",
        "profile_label": profile_name or "unknown",
        "rebalance_every": label_horizon,
        "label_horizon": label_horizon,
    }


def score_trading_dates(scores_path: Path) -> list[str]:
    try:
        scores = pd.read_parquet(scores_path, columns=["date"])
    except Exception:
        return []
    dates = pd.to_datetime(scores["date"], errors="coerce").dropna()
    if dates.empty:
        return []
    return sorted({str(pd.Timestamp(value).date()) for value in dates})


def rebalance_wait_count(trading_dates: list[str], *, last_applied_signal_date: str | None, signal_date: str) -> int:
    normalized_last = normalize_date_text(last_applied_signal_date)
    normalized_signal = normalize_date_text(signal_date) or signal_date
    if not normalized_last:
        return 0
    return sum(1 for trade_date in trading_dates if normalized_last < trade_date <= normalized_signal)


def rebalance_observed_signal_dates(
    *,
    state: dict[str, Any],
    trading_dates: list[str],
    last_applied_signal_date: str | None,
    signal_date: str,
) -> list[str]:
    normalized_last = normalize_date_text(last_applied_signal_date)
    normalized_signal = normalize_date_text(signal_date) or signal_date
    if not normalized_last:
        return []
    raw_observed = state.get("rebalance_observed_signal_dates")
    observed_values = raw_observed if isinstance(raw_observed, list) else []
    candidates = [*observed_values, *trading_dates, normalized_signal]
    normalized_dates = {
        normalized
        for value in candidates
        if (normalized := normalize_date_text(value)) and normalized_last < normalized <= normalized_signal
    }
    return sorted(normalized_dates)


def rebalance_decision(
    *,
    scores_path: Path,
    state: dict[str, Any],
    signal_date: str,
    force: bool = False,
) -> dict[str, Any]:
    profile = active_rebalance_profile(scores_path)
    rebalance_every = _positive_int(profile.get("rebalance_every"), 1)
    trading_dates = score_trading_dates(scores_path)
    last_applied_signal_date = normalize_date_text(state.get("last_applied_signal_date"))
    observed_dates = rebalance_observed_signal_dates(
        state=state,
        trading_dates=trading_dates,
        last_applied_signal_date=last_applied_signal_date,
        signal_date=signal_date,
    )
    wait_count = len(observed_dates)
    due = bool(force or rebalance_every <= 1 or not last_applied_signal_date or wait_count >= rebalance_every)
    return {
        **profile,
        "rebalance_every": rebalance_every,
        "rebalance_due": due,
        "rebalance_wait_count": wait_count,
        "rebalance_observed_signal_dates": observed_dates,
        "last_applied_signal_date": last_applied_signal_date,
    }


def load_stock_name_lookup(quant_dir: Path) -> dict[str, str]:
    for candidate in [quant_dir / "stock_list.parquet", quant_dir / "stock_registry.parquet"]:
        if not candidate.exists():
            continue
        try:
            frame = pd.read_parquet(candidate, columns=["code", "name"])
        except Exception:
            continue
        lookup: dict[str, str] = {}
        for row in frame.to_dict(orient="records"):
            code = normalize_symbol(row.get("code"))
            name = str(row.get("name") or "").strip()
            if code and name and code not in lookup:
                lookup[code] = name
        if lookup:
            return lookup
    return {}


def stock_name_for_position(symbol: str, row: dict[str, Any], lookup: dict[str, str]) -> str:
    for key in ["name", "stock_name", "security_name", "english_name"]:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return lookup.get(normalize_symbol(symbol), "")


@dataclass(frozen=True)
class SyncConfig:
    scores_path: Path
    state_dir: Path
    gateway_base_url: str
    market: str
    agent_id: str
    agent_key: str
    agent_id_header: str
    agent_key_header: str
    account_id: int | None
    top_k: int
    min_score: float
    lot_size: int
    cash_buffer_pct: float
    budget_total: float | None
    max_buy_order_qty: int
    max_sell_order_qty: int
    cancel_open_orders: bool
    sync_existing_orders: bool
    force: bool
    dry_run: bool
    paper_db_url: str | None = None
    execution_model: ExecutionModel = DEFAULT_EXECUTION_MODEL


class GatewayError(RuntimeError):
    pass


def _artifact_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def verify_immutable_model_path(project_root: Path, model: dict[str, Any]) -> Path:
    root = project_root.resolve()
    artifact_dir = (root / str(model.get("artifact_path") or "")).resolve()
    try:
        artifact_dir.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"model artifact path escapes project root: {artifact_dir}") from exc
    immutable_dir = (
        root
        / "quant_data"
        / "model_registry"
        / str(model.get("market") or "").upper()
        / str(model.get("model_version") or "")
    ).resolve()
    if artifact_dir != immutable_dir:
        raise RuntimeError(
            f"active model {model.get('model_version') or 'unknown'} uses a mutable artifact path; "
            f"expected {immutable_dir}"
        )
    return artifact_dir


def resolve_paper_model(config: SyncConfig) -> tuple[SyncConfig, dict[str, Any]]:
    """Resolve and verify one registry deployment for the entire reconciliation cycle."""
    if not config.paper_db_url:
        return config, {"source": "explicit_scores_path", "artifact_path": str(config.scores_path)}
    if psycopg is None:
        raise RuntimeError("psycopg is required to resolve the active paper model")
    with psycopg.connect(config.paper_db_url, connect_timeout=5) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select d.market, d.paper_enabled, d.revision, d.activated_at,
                       v.id, v.model_version, v.profile, v.artifact_path,
                       v.artifact_manifest, v.validation_status, v.trained_at
                from model_deployments d
                join model_versions v on v.id = d.active_model_id
                where d.market = %s
                """,
                (config.market.upper(),),
            )
            columns = [column.name for column in cursor.description]
            values = cursor.fetchone()
    if values is None:
        raise RuntimeError(f"no active {config.market.upper()} model deployment exists")
    model = dict(zip(columns, values, strict=True))
    if not bool(model["paper_enabled"]):
        raise RuntimeError(f"paper trading is disabled for {model['market']} model deployment")
    if str(model["validation_status"]) not in {"passed", "legacy_unreviewed"}:
        raise RuntimeError(
            f"active model {model['model_version']} has validation status {model['validation_status']}"
        )
    project_root = Path.cwd().resolve()
    artifact_dir = verify_immutable_model_path(project_root, model)
    manifest = model["artifact_manifest"] if isinstance(model["artifact_manifest"], dict) else {}
    for name, expected in manifest.items():
        artifact = artifact_dir / str(name)
        if not artifact.is_file() or _artifact_digest(artifact) != expected:
            raise RuntimeError(f"model artifact checksum mismatch for {name}")
    scores_path = artifact_dir / "inference_scores_latest.parquet"
    metadata_path = artifact_dir / "training_metadata.json"
    if not scores_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(f"active model {model['model_version']} is missing required artifacts")
    context = {
        "source": "postgresql_model_registry",
        "market": model["market"],
        "model_id": str(model["id"]),
        "model_version": model["model_version"],
        "profile": model["profile"],
        "artifact_path": str(model["artifact_path"]),
        "validation_status": model["validation_status"],
        "deployment_revision": int(model["revision"]),
        "activated_at": model["activated_at"],
        "trained_at": model["trained_at"],
    }
    return replace(config, scores_path=scores_path), context


class GatewayClient:
    def __init__(self, config: SyncConfig) -> None:
        self.base_url = config.gateway_base_url.rstrip("/")
        self.agent_id = config.agent_id
        self.agent_key = config.agent_key
        self.agent_id_header = config.agent_id_header
        self.agent_key_header = config.agent_key_header
        self.market = config.market
        self.account_id = config.account_id

    def _headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = {
            self.agent_id_header: self.agent_id,
            self.agent_key_header: self.agent_key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_params = {key: value for key, value in (params or {}).items() if value is not None and value != ""}
        query = f"?{urlencode(query_params)}" if query_params else ""
        data = None
        headers = self._headers()
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers = self._headers(content_type="application/json")
        request = Request(f"{self.base_url}{path}{query}", data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=10) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GatewayError(f"gateway {method} {path} failed: HTTP {exc.code} {detail}") from exc
        except URLError as exc:
            raise GatewayError(f"gateway {method} {path} failed: {exc.reason}") from exc
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise GatewayError(f"gateway {method} {path} returned invalid JSON") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def sync_agent(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/admin/sync",
            params={
                "market": self.market,
                "target_agent_id": self.agent_id,
                "account_id": self.account_id,
            },
        )

    def get_balance(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/balance", params={"market": self.market, "account_id": self.account_id}).get("balance", []))

    def get_agent_summary(self) -> dict[str, Any]:
        return dict(self._request("GET", "/v1/agents/me/summary", params={"market": self.market}).get("summary", {}))

    def get_agent_positions(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/agents/me/positions", params={"market": self.market}).get("positions", []))

    def get_agent_orders(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "/v1/agents/me/orders", params={"market": self.market}).get("orders", []))

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        remark: str,
    ) -> dict[str, Any]:
        payload = {
            "market": self.market,
            "symbol": symbol,
            "side": side,
            "order_type": "NORMAL",
            "quantity": quantity,
            "price": price,
            "remark": remark[:128],
        }
        response = self._request("POST", "/v1/orders", params={"account_id": self.account_id}, payload=payload)
        return dict(response.get("order", {}))

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        payload = {"market": self.market, "order_id": str(order_id)}
        response = self._request("POST", "/v1/orders/cancel", params={"account_id": self.account_id}, payload=payload)
        return dict(response.get("order", {}))


def load_latest_scores(config: SyncConfig) -> tuple[pd.DataFrame, str, str]:
    columns = [
        "date",
        "code",
        "exchange",
        "name",
        "industry",
        "score",
        "open",
        "close",
        "amount",
        "pct_chg",
        "pct_chg_5d",
        "pct_chg_20d",
        "change",
        "turnover",
        "turnover_ma5",
        "volume_ma5",
        "volatility_20d",
        "bias_20",
        "close_to_high_20d",
        "close_to_low_20d",
        "float_market_cap",
        "pe_ttm",
        "pb",
    ]
    try:
        scores = pd.read_parquet(config.scores_path, columns=columns)
    except Exception:
        scores = pd.read_parquet(config.scores_path)
    if scores.empty:
        raise RuntimeError("inference_scores_latest.parquet is empty")
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce")
    if scores["date"].isna().all():
        raise RuntimeError("latest score file has no valid dates")
    scores["code"] = scores["code"].astype(str).str.zfill(6)
    latest_date = pd.Timestamp(scores["date"].max()).normalize()
    latest_text = str(latest_date.date())
    latest_scores = scores[scores["date"].dt.normalize() == latest_date].copy()
    latest_scores["score"] = pd.to_numeric(latest_scores["score"], errors="coerce")
    latest_scores["close"] = pd.to_numeric(latest_scores["close"], errors="coerce")
    latest_scores = latest_scores.dropna(subset=["score", "close"])
    if exclude_st_from_model_candidates(quant_dir_for_scores_path(config.scores_path)):
        latest_scores = filter_model_candidate_rows(latest_scores, exclude_st=True)
    latest_scores = latest_scores[latest_scores["score"] >= config.min_score].sort_values("score", ascending=False).head(config.top_k)
    if latest_scores.empty:
        raise RuntimeError(f"no candidates reached score >= {config.min_score} on {latest_text}")
    latest_scores = latest_scores.reset_index(drop=True)
    latest_scores["rank"] = latest_scores.index + 1
    return latest_scores, latest_text, score_file_signature(config.scores_path)


def ensure_dirs(state_dir: Path) -> dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    return {
        "state": state_dir / "state.json",
        "targets": state_dir / "targets_latest.parquet",
        "history": state_dir / "sync_history.jsonl",
        "daily_snapshots": state_dir / DAILY_SNAPSHOTS_FILENAME,
    }


def update_state(paths: dict[str, Path], **updates: Any) -> dict[str, Any]:
    state = read_json(paths["state"])
    state.update(updates)
    state["updated_at"] = now_iso()
    write_json(paths["state"], state)
    return state


def normalize_positions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = normalize_symbol(row.get("symbol") or row.get("code"))
        if not symbol:
            continue
        normalized[symbol] = {
            **row,
            "symbol": symbol,
            "quantity": int(round(to_float(row.get("quantity")))),
            "avg_cost": to_float(row.get("avg_cost")),
            "last_price": to_float(row.get("last_price")),
            "market_value": to_float(row.get("market_value")),
            "realized_pnl": to_float(row.get("realized_pnl")),
            "unrealized_pnl": to_float(row.get("unrealized_pnl")),
        }
    return normalized


def normalize_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            {
                **row,
                "broker_order_id": str(row.get("broker_order_id") or row.get("order_id") or ""),
                "symbol": normalize_symbol(row.get("symbol") or row.get("code")),
                "order_status": normalize_status(row.get("order_status")),
                "side": str(row.get("side") or row.get("trd_side") or "").upper(),
                "quantity": int(round(to_float(row.get("quantity") or row.get("qty")))),
                "price": to_float(row.get("price")),
                "dealt_qty": to_float(row.get("dealt_qty")),
                "updated_at": str(row.get("updated_at") or row.get("create_time") or ""),
            }
        )
    return normalized


def open_position_snapshot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for row in rows:
        symbol = normalize_symbol(row.get("symbol") or row.get("code"))
        quantity = int(round(to_float(row.get("quantity") or row.get("qty"))))
        market_value = to_float(row.get("market_value") or row.get("market_val"))
        if quantity <= 0 and market_value <= 0:
            continue
        snapshots.append(
            {
                "symbol": symbol,
                "code": symbol,
                "name": row.get("name") or row.get("stock_name") or row.get("security_name") or row.get("english_name"),
                "exchange": row.get("exchange") or row.get("market"),
                "quantity": quantity,
                "last_price": to_float(row.get("last_price") or row.get("price") or row.get("current_price")),
                "avg_cost": to_float(row.get("avg_cost") or row.get("cost_price") or row.get("average_cost")),
                "market_value": market_value,
                "realized_pnl": to_float(row.get("realized_pnl")),
                "unrealized_pnl": to_float(row.get("unrealized_pnl") or row.get("pl_val")),
            }
        )
    return sorted(snapshots, key=lambda row: to_float(row.get("market_value")), reverse=True)


def order_snapshot_rows_for_trade_date(rows: list[dict[str, Any]], trade_date: str) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for row in rows:
        created_at = row.get("created_at") or row.get("create_time") or row.get("updated_at")
        updated_at = row.get("updated_at") or row.get("create_time") or row.get("created_at")
        if trade_date_text(created_at or updated_at) != trade_date:
            continue
        snapshots.append(snapshot_order_event(row))
    return sorted(snapshots, key=lambda row: str(row.get("created_at") or row.get("updated_at") or ""), reverse=True)


def daily_snapshot_summary(
    *,
    balance_metrics: dict[str, Any],
    live_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "total_assets": live_summary.get("total_assets") or balance_metrics.get("total_assets"),
        "cash": balance_metrics.get("cash"),
        "buying_power": balance_metrics.get("power"),
        "market_value": live_summary.get("market_value"),
        "realized_pnl": live_summary.get("realized_pnl"),
        "unrealized_pnl": live_summary.get("unrealized_pnl"),
        "total_pnl": live_summary.get("total_pnl"),
        "currency": balance_metrics.get("currency"),
    }


def write_daily_snapshot(
    path: Path,
    *,
    trade_date: str,
    balance_metrics: dict[str, Any],
    live_summary: dict[str, Any],
    positions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> None:
    snapshots = read_json(path)
    if not isinstance(snapshots, dict):
        snapshots = {}
    position_rows = open_position_snapshot_rows(positions)
    order_rows = order_snapshot_rows_for_trade_date(orders, trade_date)
    snapshots[trade_date] = {
        "trade_date": trade_date,
        "generated_at": now_iso(),
        "summary": daily_snapshot_summary(balance_metrics=balance_metrics, live_summary=live_summary),
        "positions_rows": len(position_rows),
        "positions": position_rows,
        "orders_rows": len(order_rows),
        "orders": order_rows,
        "positions_snapshot_available": True,
    }
    write_json(path, snapshots)


def safe_write_daily_snapshot(
    paths: dict[str, Path],
    *,
    balance_metrics: dict[str, Any],
    live_summary: dict[str, Any],
    positions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> None:
    try:
        write_daily_snapshot(
            paths["daily_snapshots"],
            trade_date=trade_date_text(),
            balance_metrics=balance_metrics,
            live_summary=live_summary,
            positions=positions,
            orders=orders,
        )
    except Exception as exc:  # pragma: no cover - snapshot failures must not block order safety
        print(f"failed to write daily paper snapshot: {exc}", file=sys.stderr, flush=True)


def snapshot_order_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "broker_order_id": str(row.get("broker_order_id") or row.get("order_id") or ""),
        "symbol": normalize_symbol(row.get("symbol") or row.get("code")),
        "side": str(row.get("side") or row.get("trd_side") or "").upper(),
        "order_status": normalize_status(row.get("order_status")),
        "quantity": int(round(to_float(row.get("quantity") or row.get("qty")))),
        "price": to_float(row.get("price")),
        "dealt_qty": to_float(row.get("dealt_qty")),
        "dealt_avg_price": to_float(row.get("dealt_avg_price")),
        "remark": str(row.get("remark") or ""),
        "created_at": str(row.get("created_at") or row.get("create_time") or ""),
        "updated_at": str(row.get("updated_at") or row.get("create_time") or row.get("created_at") or ""),
    }


def snapshot_skipped_order(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": normalize_symbol(row.get("symbol")),
        "side": str(row.get("side") or "").upper(),
        "quantity": int(round(to_float(row.get("quantity")))),
        "attempted_prices": [round_price(to_float(price)) for price in list(row.get("attempted_prices") or []) if to_float(price) > 0],
        "error": str(row.get("error") or ""),
    }


def determine_total_capital(
    config: SyncConfig,
    *,
    balance_metrics: dict[str, float | str | None],
    current_market_value: float,
) -> float:
    if config.budget_total is not None and config.budget_total > 0:
        return config.budget_total
    total_assets = to_float(balance_metrics.get("total_assets"))
    if total_assets > 0:
        return total_assets
    cash = max(to_float(balance_metrics.get("power")), to_float(balance_metrics.get("cash")))
    return cash + current_market_value


def buy_capacity(
    config: SyncConfig,
    *,
    balance_metrics: dict[str, float | str | None],
    current_market_value: float,
    planned_sale_notional: float,
    planned_sale_fee: float = 0.0,
) -> float:
    sale_proceeds = clamp_non_negative(planned_sale_notional - planned_sale_fee)
    if config.budget_total is not None and config.budget_total > 0:
        remaining = config.budget_total - current_market_value + sale_proceeds
        return clamp_non_negative(remaining)
    power = max(to_float(balance_metrics.get("power")), to_float(balance_metrics.get("cash")))
    return clamp_non_negative(power + sale_proceeds)


def compute_order_quantity(
    *,
    side: str,
    raw_quantity: int,
    lot_size: int,
    full_exit: bool = False,
) -> int:
    quantity = int(max(raw_quantity, 0))
    if quantity <= 0:
        return 0
    if side == "SELL" and full_exit:
        return quantity
    lots = quantity // max(lot_size, 1)
    return lots * max(lot_size, 1)


def apply_optional_quantity_cap(quantity: int, max_quantity: int | None) -> int:
    normalized = int(max(quantity, 0))
    if normalized <= 0:
        return 0
    if max_quantity is None:
        return normalized
    cap = int(max_quantity or 0)
    if cap <= 0:
        return 0
    return min(normalized, cap)


def estimate_order_fee(side: str, quantity: int, price: float) -> float:
    return transaction_fee(side, max(int(quantity), 0) * max(float(price), 0.0), DEFAULT_FEE_MODEL)


def compute_affordable_buy_quantity(*, cash_available: float, price: float, lot_size: int) -> int:
    if cash_available <= 0 or price <= 0:
        return 0
    lot = max(int(lot_size), 1)
    low = 0
    high = int(cash_available / price) // lot
    affordable_lots = 0
    while low <= high:
        mid = (low + high) // 2
        quantity = mid * lot
        notional = quantity * price
        total_cost = notional + transaction_fee("BUY", notional, DEFAULT_FEE_MODEL)
        if total_cost <= cash_available:
            affordable_lots = mid
            low = mid + 1
        else:
            high = mid - 1
    return affordable_lots * lot


def position_quantity(row: dict[str, Any]) -> int:
    return int(round(to_float(row.get("quantity"))))


def planning_position_for_symbol(
    symbol: str,
    positions: dict[str, dict[str, Any]],
    managed_positions: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    broker = positions.get(symbol, {})
    if managed_positions is None:
        return broker

    managed = managed_positions.get(symbol, {})
    broker_qty = position_quantity(broker)
    managed_qty = int(round(to_float(managed.get("quantity"))))
    quantity = min(max(managed_qty, 0), max(broker_qty, 0))
    last_price = (
        to_float(broker.get("last_price"))
        or to_float(managed.get("last_price"))
        or to_float(broker.get("avg_cost"))
        or to_float(managed.get("avg_cost"))
    )
    avg_cost = to_float(managed.get("avg_cost")) or to_float(broker.get("avg_cost"))
    return {
        **broker,
        "symbol": symbol,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "last_price": last_price,
        "market_value": quantity * last_price,
        "realized_pnl": to_float(managed.get("realized_pnl")) or to_float(broker.get("realized_pnl")),
        "unrealized_pnl": to_float(broker.get("unrealized_pnl")),
        "managed_quantity": managed_qty,
        "broker_quantity": broker_qty,
    }


def planning_market_value(
    positions: dict[str, dict[str, Any]],
    managed_positions: dict[str, dict[str, Any]] | None,
) -> float:
    if managed_positions is None:
        return sum(to_float(item.get("market_value")) for item in positions.values())
    return sum(
        to_float(planning_position_for_symbol(symbol, positions, managed_positions).get("market_value"))
        for symbol in managed_positions
    )


def build_plan(
    config: SyncConfig,
    *,
    latest_scores: pd.DataFrame,
    positions: dict[str, dict[str, Any]],
    balance_metrics: dict[str, float | str | None],
    managed_positions: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    managed_positions_enabled = managed_positions is not None
    current_market_value = planning_market_value(positions, managed_positions)
    total_capital = determine_total_capital(config, balance_metrics=balance_metrics, current_market_value=current_market_value)
    investable_capital = clamp_non_negative(total_capital * (1.0 - config.cash_buffer_pct))
    target_count = int(len(latest_scores))
    target_value = investable_capital / target_count if target_count else 0.0

    score_lookup: dict[str, dict[str, Any]] = {}
    plan_rows: list[dict[str, Any]] = []
    target_symbols: list[str] = []
    manual_st_positions_ignored: list[str] = []
    manual_position_symbols_ignored = sorted(
        symbol
        for symbol, row in positions.items()
        if managed_positions_enabled
        and position_quantity(row) > 0
        and int(round(to_float((managed_positions or {}).get(symbol, {}).get("quantity")))) <= 0
    )
    exclude_st = exclude_st_from_model_candidates(quant_dir_for_scores_path(config.scores_path))
    stock_name_lookup = load_stock_name_lookup(quant_dir_for_scores_path(config.scores_path)) if exclude_st else {}
    for _, row in latest_scores.iterrows():
        symbol = normalize_symbol(row["code"])
        target_symbols.append(symbol)
        score_row = row.to_dict()
        score_lookup[symbol] = score_row
        current = planning_position_for_symbol(symbol, positions, managed_positions)
        current_qty = position_quantity(current)
        close_price = to_float(row["close"])
        reference_amount = to_float(row.get("amount"))
        previous_close = previous_close_from_row(row.to_dict())
        buy_price = build_reference_price(close_price)
        sell_price = build_reference_price(close_price)
        theoretical_qty = int(target_value / buy_price) if buy_price > 0 else 0
        target_qty = compute_order_quantity(side="BUY", raw_quantity=theoretical_qty, lot_size=config.lot_size)
        liquidity_cap = liquidity_cap_notional(reference_amount, config.execution_model)
        plan_rows.append(
            {
                "signal_date": normalize_date_text(row["date"]),
                "rank": int(score_row["rank"]),
                "code": symbol,
                "exchange": str(row.get("exchange") or ""),
                "name": str(row.get("name") or ""),
                "industry": str(row.get("industry") or ""),
                "score": to_float(row.get("score")),
                "close": close_price,
                "open": to_float(row.get("open")),
                "amount": reference_amount,
                "pct_chg": to_float(row.get("pct_chg")),
                "pct_chg_5d": to_float(row.get("pct_chg_5d")),
                "pct_chg_20d": to_float(row.get("pct_chg_20d")),
                "change": to_float(row.get("change")),
                "turnover": to_float(row.get("turnover")),
                "turnover_ma5": to_float(row.get("turnover_ma5")),
                "volume_ma5": to_float(row.get("volume_ma5")),
                "volatility_20d": to_float(row.get("volatility_20d")),
                "bias_20": to_float(row.get("bias_20")),
                "close_to_high_20d": to_float(row.get("close_to_high_20d")),
                "close_to_low_20d": to_float(row.get("close_to_low_20d")),
                "float_market_cap": to_float(row.get("float_market_cap")),
                "pe_ttm": to_float(row.get("pe_ttm")),
                "pb": to_float(row.get("pb")),
                "previous_close": previous_close,
                "liquidity_cap_notional": liquidity_cap,
                "buy_limit_price": buy_price,
                "sell_limit_price": sell_price,
                "target_weight": 1.0 / target_count if target_count else 0.0,
                "target_value": target_value,
                "target_qty": target_qty,
                "current_qty": current_qty,
                "delta_qty": target_qty - current_qty,
                "current_market_value": to_float(current.get("market_value")),
                "current_avg_cost": to_float(current.get("avg_cost")),
                "current_last_price": to_float(current.get("last_price")),
                "reason": "ranked_target",
            }
        )

    exit_symbols = (managed_positions or positions).keys()
    for symbol in exit_symbols:
        if symbol in score_lookup:
            continue
        current = planning_position_for_symbol(symbol, positions, managed_positions)
        current_qty = position_quantity(current)
        if current_qty <= 0:
            continue
        current_name = stock_name_for_position(symbol, current, stock_name_lookup)
        if exclude_st and is_st_stock_name(current_name):
            manual_st_positions_ignored.append(symbol)
            continue
        base_price = to_float(current.get("last_price")) or to_float(current.get("avg_cost"))
        plan_rows.append(
            {
                "signal_date": normalize_date_text(latest_scores["date"].iloc[0]),
                "rank": None,
                "code": symbol,
                "exchange": "",
                "name": str(current.get("symbol") or symbol),
                "industry": "",
                "score": None,
                "close": base_price,
                "open": None,
                "amount": None,
                "pct_chg": None,
                "pct_chg_5d": None,
                "pct_chg_20d": None,
                "change": None,
                "turnover": None,
                "turnover_ma5": None,
                "volume_ma5": None,
                "volatility_20d": None,
                "bias_20": None,
                "close_to_high_20d": None,
                "close_to_low_20d": None,
                "float_market_cap": None,
                "pe_ttm": None,
                "pb": None,
                "previous_close": None,
                "liquidity_cap_notional": None,
                "buy_limit_price": build_reference_price(base_price),
                "sell_limit_price": build_reference_price(base_price),
                "target_weight": 0.0,
                "target_value": 0.0,
                "target_qty": 0,
                "current_qty": current_qty,
                "delta_qty": -current_qty,
                "current_market_value": to_float(current.get("market_value")),
                "current_avg_cost": to_float(current.get("avg_cost")),
                "current_last_price": to_float(current.get("last_price")),
                "reason": "exit_non_target",
            }
        )

    plan = pd.DataFrame(plan_rows)
    if plan.empty:
        raise RuntimeError("rebalance plan is empty")

    plan["sell_qty"] = plan["delta_qty"].apply(lambda value: abs(int(value)) if value < 0 else 0)
    plan["buy_qty"] = plan["delta_qty"].apply(lambda value: int(value) if value > 0 else 0)
    plan["action"] = "HOLD"
    plan.loc[plan["sell_qty"] > 0, "action"] = "SELL"
    plan.loc[(plan["sell_qty"] == 0) & (plan["buy_qty"] > 0), "action"] = "BUY"

    plan["sell_order_qty"] = plan.apply(
        lambda row: compute_order_quantity(
            side="SELL",
            raw_quantity=int(row["sell_qty"]),
            lot_size=config.lot_size,
            full_exit=int(row["target_qty"]) == 0 and int(row["current_qty"]) > 0,
        ),
        axis=1,
    )
    plan["buy_order_qty"] = plan.apply(
        lambda row: compute_order_quantity(side="BUY", raw_quantity=int(row["buy_qty"]), lot_size=config.lot_size),
        axis=1,
    )

    plan["estimated_order_notional"] = 0.0
    sell_mask = plan["sell_order_qty"] > 0
    buy_mask = plan["buy_order_qty"] > 0
    plan.loc[sell_mask, "estimated_order_notional"] = plan.loc[sell_mask, "sell_order_qty"] * plan.loc[sell_mask, "sell_limit_price"]
    plan.loc[buy_mask, "estimated_order_notional"] = plan.loc[buy_mask, "buy_order_qty"] * plan.loc[buy_mask, "buy_limit_price"]
    plan["estimated_order_fee"] = 0.0
    plan.loc[sell_mask, "estimated_order_fee"] = plan.loc[sell_mask].apply(
        lambda row: estimate_order_fee("SELL", int(row["sell_order_qty"]), to_float(row["sell_limit_price"])),
        axis=1,
    )
    plan.loc[buy_mask, "estimated_order_fee"] = plan.loc[buy_mask].apply(
        lambda row: estimate_order_fee("BUY", int(row["buy_order_qty"]), to_float(row["buy_limit_price"])),
        axis=1,
    )

    planned_sale_notional = float(plan.loc[sell_mask, "estimated_order_notional"].sum())
    planned_sale_fee = float(plan.loc[sell_mask, "estimated_order_fee"].sum())
    planned_sale_proceeds = clamp_non_negative(planned_sale_notional - planned_sale_fee)
    remaining_buy_capacity = buy_capacity(
        config,
        balance_metrics=balance_metrics,
        current_market_value=current_market_value,
        planned_sale_notional=planned_sale_notional,
        planned_sale_fee=planned_sale_fee,
    )

    for index, row in plan.sort_values(["rank", "score"], ascending=[True, False], na_position="last").iterrows():
        buy_qty = int(row["buy_order_qty"])
        if buy_qty <= 0:
            continue
        price = to_float(row["buy_limit_price"])
        if price <= 0:
            plan.at[index, "buy_order_qty"] = 0
            plan.at[index, "estimated_order_notional"] = 0.0
            plan.at[index, "estimated_order_fee"] = 0.0
            plan.at[index, "action"] = "SKIP_INVALID_PRICE"
            continue
        affordable_qty = compute_affordable_buy_quantity(
            cash_available=remaining_buy_capacity,
            price=price,
            lot_size=config.lot_size,
        )
        actual_qty = min(apply_optional_quantity_cap(buy_qty, config.max_buy_order_qty), affordable_qty)
        if actual_qty <= 0:
            plan.at[index, "buy_order_qty"] = 0
            plan.at[index, "estimated_order_notional"] = 0.0
            plan.at[index, "estimated_order_fee"] = 0.0
            plan.at[index, "action"] = "SKIP_NO_CASH"
            continue
        order_notional = actual_qty * price
        skip_reason = buy_liquidity_skip_reason(
            amount=row.get("amount"),
            order_notional=order_notional,
            model=config.execution_model,
        )
        if skip_reason is None and near_price_limit(
            side="BUY",
            price=price,
            previous_close=to_float(row.get("previous_close")),
            symbol=row.get("code"),
            name=row.get("name"),
            model=config.execution_model,
        ):
            skip_reason = "SKIP_LIMIT_UP"
        if skip_reason is not None:
            plan.at[index, "buy_order_qty"] = 0
            plan.at[index, "estimated_order_notional"] = 0.0
            plan.at[index, "estimated_order_fee"] = 0.0
            plan.at[index, "action"] = skip_reason
            continue
        plan.at[index, "buy_order_qty"] = actual_qty
        order_fee = transaction_fee("BUY", order_notional, DEFAULT_FEE_MODEL)
        plan.at[index, "estimated_order_notional"] = order_notional
        plan.at[index, "estimated_order_fee"] = order_fee
        remaining_buy_capacity -= order_notional + order_fee

    sell_mask = plan["sell_order_qty"] > 0
    buy_mask = plan["buy_order_qty"] > 0
    estimated_buy_notional = float(plan.loc[buy_mask, "estimated_order_notional"].sum())
    estimated_buy_fee = float(plan.loc[buy_mask, "estimated_order_fee"].sum())
    estimated_order_fee = float(plan["estimated_order_fee"].sum())

    summary = {
        "target_symbols": target_symbols,
        "target_count": target_count,
        "current_market_value": current_market_value,
        "total_capital": total_capital,
        "investable_capital": investable_capital,
        "planned_sale_notional": planned_sale_notional,
        "planned_sale_fee": planned_sale_fee,
        "planned_sale_proceeds": planned_sale_proceeds,
        "planned_buy_notional": estimated_buy_notional,
        "planned_buy_fee": estimated_buy_fee,
        "estimated_order_fee": estimated_order_fee,
        "fee_model": {
            "commission_rate": DEFAULT_FEE_MODEL.commission_rate,
            "min_commission": DEFAULT_FEE_MODEL.min_commission,
            "platform_fee": DEFAULT_FEE_MODEL.platform_fee,
            "tiny_fee_rate": DEFAULT_FEE_MODEL.tiny_fee_rate,
            "sell_stamp_duty_rate": DEFAULT_FEE_MODEL.sell_stamp_duty_rate,
        },
        "execution_model": execution_model_snapshot(config.execution_model),
        "buy_capacity": buy_capacity(
            config,
            balance_metrics=balance_metrics,
            current_market_value=current_market_value,
            planned_sale_notional=planned_sale_notional,
            planned_sale_fee=planned_sale_fee,
        ),
        "sell_order_count": int((plan["sell_order_qty"] > 0).sum()),
        "buy_order_count": int((plan["buy_order_qty"] > 0).sum()),
        "skip_count": int(plan["action"].astype(str).str.startswith("SKIP_").sum()),
        "manual_st_positions_ignored": manual_st_positions_ignored,
        "manual_position_symbols_ignored": manual_position_symbols_ignored,
        "managed_position_source": "paper_managed_positions" if managed_positions_enabled else "broker_positions",
        "managed_position_count": len([row for row in (managed_positions or {}).values() if to_float(row.get("quantity")) > 0]),
    }
    return plan, summary


def persist_targets(paths: dict[str, Path], plan: pd.DataFrame) -> None:
    ordered_cols = [
        "signal_date",
        "rank",
        "code",
        "exchange",
        "name",
        "industry",
        "score",
        "open",
        "close",
        "amount",
        "pct_chg",
        "pct_chg_5d",
        "pct_chg_20d",
        "change",
        "turnover",
        "turnover_ma5",
        "volume_ma5",
        "volatility_20d",
        "bias_20",
        "close_to_high_20d",
        "close_to_low_20d",
        "float_market_cap",
        "pe_ttm",
        "pb",
        "previous_close",
        "liquidity_cap_notional",
        "buy_limit_price",
        "sell_limit_price",
        "target_weight",
        "target_value",
        "target_qty",
        "current_qty",
        "delta_qty",
        "sell_order_qty",
        "buy_order_qty",
        "action",
        "sent_order_id",
        "sent_status",
        "sent_price",
        "sent_reference_price",
        "sent_error",
        "estimated_order_notional",
        "estimated_order_fee",
        "reason",
        "current_market_value",
        "current_avg_cost",
        "current_last_price",
    ]
    available_cols = [column for column in ordered_cols if column in plan.columns]
    actionable = pd.Series(False, index=plan.index)
    for column in [
        "target_qty",
        "current_qty",
        "delta_qty",
        "sell_order_qty",
        "buy_order_qty",
        "estimated_order_notional",
        "estimated_order_fee",
        "current_market_value",
    ]:
        if column in plan.columns:
            actionable |= pd.to_numeric(plan[column], errors="coerce").fillna(0).abs() > 0
    if "action" in plan.columns:
        action = plan["action"].astype("string").fillna("").str.strip().str.upper()
        actionable |= action.ne("") & action.ne("HOLD")
    for column in ["sent_order_id", "sent_status", "sent_error"]:
        if column in plan.columns:
            actionable |= plan[column].astype("string").fillna("").str.strip().ne("")
    persisted = plan.loc[actionable, available_cols]
    if not persisted.empty:
        persisted = persisted.sort_values(["rank", "score"], ascending=[True, False], na_position="last")
    persisted.to_parquet(
        paths["targets"],
        index=False,
    )


def target_snapshot_rows(plan: pd.DataFrame) -> list[dict[str, Any]]:
    if plan.empty:
        return []
    columns = [
        "signal_date",
        "rank",
        "code",
        "exchange",
        "name",
        "industry",
        "score",
        "close",
        "open",
        "amount",
        "pct_chg",
        "pct_chg_5d",
        "pct_chg_20d",
        "turnover",
        "turnover_ma5",
        "volume_ma5",
        "volatility_20d",
        "bias_20",
        "close_to_high_20d",
        "close_to_low_20d",
        "float_market_cap",
        "pe_ttm",
        "pb",
        "target_weight",
        "target_qty",
        "current_qty",
        "delta_qty",
        "buy_order_qty",
        "sell_order_qty",
        "action",
        "reason",
        "estimated_order_notional",
        "estimated_order_fee",
        "sent_order_id",
        "sent_status",
        "sent_price",
        "sent_error",
    ]
    available = [column for column in columns if column in plan.columns]
    snapshot = plan.copy()
    if "rank" in snapshot.columns:
        ranked = pd.to_numeric(snapshot["rank"], errors="coerce").notna()
        actionable = pd.Series(False, index=snapshot.index)
        for column in ["current_qty", "delta_qty", "buy_order_qty", "sell_order_qty"]:
            if column in snapshot.columns:
                actionable |= pd.to_numeric(snapshot[column], errors="coerce").fillna(0).abs() > 0
        snapshot = snapshot[ranked | actionable]
    if not snapshot.empty:
        snapshot = snapshot.sort_values(["rank", "score"], ascending=[True, False], na_position="last")
    return json.loads(json.dumps(snapshot[available].to_dict(orient="records"), ensure_ascii=False, default=json_default))


def db_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return None
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): db_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [db_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return db_jsonable(value.item())
        except Exception:
            return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def db_jsonb(value: Any) -> Any:
    return Jsonb(db_jsonable(value)) if Jsonb is not None else json.dumps(db_jsonable(value), ensure_ascii=False)


def db_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def db_float(value: Any) -> float | None:
    if value in (None, "", "NaN", "nan"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def db_int(value: Any) -> int | None:
    number = db_float(value)
    return int(number) if number is not None else None


def ensure_paper_history_schema(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            create table if not exists paper_rebalance_runs (
              id bigserial primary key,
              agent_id text not null,
              market text not null,
              score_signal_date date,
              recorded_at timestamptz not null,
              status text,
              message text,
              last_score_signature text,
              gateway_healthy boolean,
              profile_name text,
              profile_label text,
              rebalance_every integer,
              rebalance_due boolean,
              rebalance_wait_count integer,
              plan_summary jsonb not null default '{}'::jsonb,
              balance_metrics jsonb not null default '{}'::jsonb,
              live_summary jsonb not null default '{}'::jsonb,
              active_order_count integer,
              position_count integer,
              placed_order_ids text[] not null default '{}',
              cancelled_order_ids text[] not null default '{}',
              skipped_symbols text[] not null default '{}',
              placed_orders jsonb not null default '[]'::jsonb,
              cancelled_orders jsonb not null default '[]'::jsonb,
              skipped_orders jsonb not null default '[]'::jsonb,
              created_at timestamptz not null default now()
            )
            """
        )
        cur.execute(
            """
            create table if not exists paper_rebalance_targets (
              id bigserial primary key,
              run_id bigint not null references paper_rebalance_runs(id) on delete cascade,
              agent_id text not null,
              market text not null,
              score_signal_date date,
              recorded_at timestamptz not null,
              code text not null,
              exchange text,
              name text,
              industry text,
              rank integer,
              score double precision,
              open double precision,
              close double precision,
              amount double precision,
              pct_chg double precision,
              pct_chg_5d double precision,
              pct_chg_20d double precision,
              turnover double precision,
              turnover_ma5 double precision,
              volume_ma5 double precision,
              volatility_20d double precision,
              bias_20 double precision,
              close_to_high_20d double precision,
              close_to_low_20d double precision,
              float_market_cap double precision,
              pe_ttm double precision,
              pb double precision,
              target_weight double precision,
              target_qty integer,
              current_qty integer,
              delta_qty integer,
              buy_order_qty integer,
              sell_order_qty integer,
              action text,
              reason text,
              estimated_order_notional double precision,
              estimated_order_fee double precision,
              sent_order_id text,
              sent_status text,
              sent_price double precision,
              sent_error text,
              target_snapshot jsonb not null default '{}'::jsonb,
              created_at timestamptz not null default now()
            )
            """
        )
        cur.execute(
            "create index if not exists idx_paper_rebalance_runs_agent_date "
            "on paper_rebalance_runs(agent_id, market, score_signal_date desc, recorded_at desc)"
        )
        cur.execute(
            "create index if not exists idx_paper_rebalance_targets_symbol_date "
            "on paper_rebalance_targets(agent_id, market, code, score_signal_date desc, rank)"
        )
        cur.execute(
            "create index if not exists idx_paper_rebalance_targets_run_rank "
            "on paper_rebalance_targets(run_id, rank)"
        )
        cur.execute(
            """
            create table if not exists paper_managed_positions (
              agent_id text not null,
              market text not null,
              strategy_id text not null,
              symbol text not null,
              quantity double precision not null default 0,
              avg_cost double precision not null default 0,
              cost_basis double precision not null default 0,
              realized_pnl double precision not null default 0,
              buy_qty double precision not null default 0,
              sell_qty double precision not null default 0,
              last_price double precision not null default 0,
              last_fill_at timestamptz,
              source_fill_count integer not null default 0,
              updated_at timestamptz not null default now(),
              primary key (agent_id, market, strategy_id, symbol)
            )
            """
        )
        cur.execute(
            "create index if not exists idx_paper_managed_positions_open "
            "on paper_managed_positions(agent_id, market, strategy_id, quantity desc)"
        )


def build_managed_positions_from_fills(fills: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    positions: dict[str, dict[str, Any]] = {}
    for fill in fills:
        symbol = normalize_symbol(fill.get("symbol"))
        if not symbol:
            continue
        position = positions.setdefault(
            symbol,
            {
                "symbol": symbol,
                "quantity": 0.0,
                "cost_basis": 0.0,
                "avg_cost": 0.0,
                "realized_pnl": 0.0,
                "buy_qty": 0.0,
                "sell_qty": 0.0,
                "last_price": 0.0,
                "last_fill_at": None,
                "source_fill_count": 0,
            },
        )
        side = str(fill.get("side") or "").upper()
        quantity = max(to_float(fill.get("quantity")), 0.0)
        price = max(to_float(fill.get("price")), 0.0)
        notional = max(to_float(fill.get("notional")) or quantity * price, 0.0)
        current_qty = to_float(position.get("quantity"))
        cost_basis = to_float(position.get("cost_basis"))
        avg_before = cost_basis / current_qty if current_qty > 0 else 0.0

        if side == "BUY":
            current_qty += quantity
            cost_basis += notional
            position["buy_qty"] = to_float(position.get("buy_qty")) + quantity
        elif side == "SELL":
            matched_qty = min(current_qty, quantity)
            position["realized_pnl"] = to_float(position.get("realized_pnl")) + (price - avg_before) * matched_qty
            current_qty -= matched_qty
            cost_basis -= avg_before * matched_qty
            position["sell_qty"] = to_float(position.get("sell_qty")) + quantity
            if current_qty <= 1e-9:
                current_qty = 0.0
                cost_basis = 0.0

        position["quantity"] = current_qty
        position["cost_basis"] = cost_basis
        position["avg_cost"] = cost_basis / current_qty if current_qty > 0 else 0.0
        position["last_price"] = price
        position["last_fill_at"] = fill.get("created_at")
        position["source_fill_count"] = int(position.get("source_fill_count") or 0) + 1
    return positions


def refresh_managed_positions_from_db(config: SyncConfig) -> dict[str, dict[str, Any]] | None:
    if not config.paper_db_url:
        return None
    if psycopg is None or Jsonb is None:
        raise RuntimeError("PAPER_DB_URL is configured but psycopg is unavailable; cannot load managed positions")

    with psycopg.connect(config.paper_db_url, connect_timeout=5) as conn:
        ensure_paper_history_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                  f.created_at, f.broker_order_id, f.market, f.symbol, f.side,
                  f.quantity, f.price, f.notional
                from agent_fills f
                left join agent_order_snapshots o
                  on o.agent_id = f.agent_id
                 and o.market = f.market
                 and o.broker_order_id = f.broker_order_id
                where f.agent_id = %s
                  and f.market = %s
                  and (
                    exists (
                      select 1
                      from paper_rebalance_targets t
                      where t.agent_id = f.agent_id
                        and t.market = f.market
                        and t.sent_order_id = f.broker_order_id
                    )
                    or coalesce(o.remark, '') like %s
                  )
                order by f.created_at asc, f.id asc
                """,
                [config.agent_id, config.market, f"{QUANT_ORDER_REMARK_PREFIX}%"],
            )
            columns = [desc[0] for desc in cur.description]
            fills = [dict(zip(columns, row)) for row in cur.fetchall()]
            managed_positions = build_managed_positions_from_fills(fills)

            cur.execute(
                """
                delete from paper_managed_positions
                where agent_id = %s and market = %s and strategy_id = %s
                """,
                [config.agent_id, config.market, DEFAULT_STRATEGY_ID],
            )
            for symbol, position in managed_positions.items():
                cur.execute(
                    """
                    insert into paper_managed_positions (
                      agent_id, market, strategy_id, symbol, quantity, avg_cost,
                      cost_basis, realized_pnl, buy_qty, sell_qty, last_price,
                      last_fill_at, source_fill_count, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    """,
                    [
                        config.agent_id,
                        config.market,
                        DEFAULT_STRATEGY_ID,
                        symbol,
                        to_float(position.get("quantity")),
                        to_float(position.get("avg_cost")),
                        to_float(position.get("cost_basis")),
                        to_float(position.get("realized_pnl")),
                        to_float(position.get("buy_qty")),
                        to_float(position.get("sell_qty")),
                        to_float(position.get("last_price")),
                        position.get("last_fill_at"),
                        int(position.get("source_fill_count") or 0),
                    ],
                )
        conn.commit()

    return {
        symbol: position
        for symbol, position in managed_positions.items()
        if to_float(position.get("quantity")) > 1e-9
    }


def persist_rebalance_result_to_db(config: SyncConfig, result: dict[str, Any]) -> None:
    if not config.paper_db_url:
        return
    if psycopg is None or Jsonb is None:
        print("PAPER_DB_URL is configured but psycopg is unavailable; skipping DB persistence", file=sys.stderr, flush=True)
        return

    plan = result.get("plan_summary") if isinstance(result.get("plan_summary"), dict) else {}
    recorded_at = result.get("recorded_at") or now_iso()
    score_signal_date = normalize_date_text(result.get("score_signal_date"))
    with psycopg.connect(config.paper_db_url, connect_timeout=5) as conn:
        ensure_paper_history_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into paper_rebalance_runs (
                  agent_id, market, score_signal_date, recorded_at, status, message,
                  last_score_signature, gateway_healthy, profile_name, profile_label,
                  rebalance_every, rebalance_due, rebalance_wait_count,
                  plan_summary, balance_metrics, live_summary,
                  active_order_count, position_count,
                  placed_order_ids, cancelled_order_ids, skipped_symbols,
                  placed_orders, cancelled_orders, skipped_orders
                )
                values (
                  %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s,
                  %s, %s, %s,
                  %s, %s, %s,
                  %s, %s,
                  %s, %s, %s,
                  %s, %s, %s
                )
                returning id
                """,
                [
                    config.agent_id,
                    config.market,
                    score_signal_date,
                    recorded_at,
                    db_text(result.get("status")),
                    db_text(result.get("message")),
                    db_text(result.get("last_score_signature")),
                    bool(result.get("gateway_healthy")) if result.get("gateway_healthy") is not None else None,
                    db_text(plan.get("profile_name")),
                    db_text(plan.get("profile_label")),
                    db_int(plan.get("rebalance_every")),
                    bool(plan.get("rebalance_due")) if plan.get("rebalance_due") is not None else None,
                    db_int(plan.get("rebalance_wait_count")),
                    db_jsonb(plan),
                    db_jsonb(result.get("balance_metrics") or {}),
                    db_jsonb(result.get("live_summary") or {}),
                    db_int(result.get("active_order_count")),
                    db_int(result.get("position_count")),
                    [str(item) for item in result.get("placed_order_ids") or [] if str(item)],
                    [str(item) for item in result.get("cancelled_order_ids") or [] if str(item)],
                    [str(item) for item in result.get("skipped_symbols") or [] if str(item)],
                    db_jsonb(result.get("placed_orders") or []),
                    db_jsonb(result.get("cancelled_orders") or []),
                    db_jsonb(result.get("skipped_orders") or []),
                ],
            )
            run_id = cur.fetchone()[0]
            for target in result.get("target_snapshot") or []:
                if not isinstance(target, dict):
                    continue
                code = normalize_symbol(target.get("code") or target.get("symbol"))
                if not code:
                    continue
                cur.execute(
                    """
                    insert into paper_rebalance_targets (
                      run_id, agent_id, market, score_signal_date, recorded_at,
                      code, exchange, name, industry, rank, score, open, close, amount,
                      pct_chg, pct_chg_5d, pct_chg_20d, turnover, turnover_ma5,
                      volume_ma5, volatility_20d, bias_20, close_to_high_20d,
                      close_to_low_20d, float_market_cap, pe_ttm, pb,
                      target_weight, target_qty, current_qty, delta_qty, buy_order_qty,
                      sell_order_qty, action, reason, estimated_order_notional,
                      estimated_order_fee, sent_order_id, sent_status, sent_price,
                      sent_error, target_snapshot
                    )
                    values (
                      %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s,
                      %s, %s, %s, %s,
                      %s, %s, %s, %s,
                      %s, %s, %s, %s, %s,
                      %s, %s, %s, %s,
                      %s, %s, %s, %s,
                      %s, %s
                    )
                    """,
                    [
                        run_id,
                        config.agent_id,
                        config.market,
                        score_signal_date,
                        recorded_at,
                        code,
                        db_text(target.get("exchange")),
                        db_text(target.get("name")),
                        db_text(target.get("industry")),
                        db_int(target.get("rank")),
                        db_float(target.get("score")),
                        db_float(target.get("open")),
                        db_float(target.get("close")),
                        db_float(target.get("amount")),
                        db_float(target.get("pct_chg")),
                        db_float(target.get("pct_chg_5d")),
                        db_float(target.get("pct_chg_20d")),
                        db_float(target.get("turnover")),
                        db_float(target.get("turnover_ma5")),
                        db_float(target.get("volume_ma5")),
                        db_float(target.get("volatility_20d")),
                        db_float(target.get("bias_20")),
                        db_float(target.get("close_to_high_20d")),
                        db_float(target.get("close_to_low_20d")),
                        db_float(target.get("float_market_cap")),
                        db_float(target.get("pe_ttm")),
                        db_float(target.get("pb")),
                        db_float(target.get("target_weight")),
                        db_int(target.get("target_qty")),
                        db_int(target.get("current_qty")),
                        db_int(target.get("delta_qty")),
                        db_int(target.get("buy_order_qty")),
                        db_int(target.get("sell_order_qty")),
                        db_text(target.get("action")),
                        db_text(target.get("reason")),
                        db_float(target.get("estimated_order_notional")),
                        db_float(target.get("estimated_order_fee")),
                        db_text(target.get("sent_order_id")),
                        db_text(target.get("sent_status")),
                        db_float(target.get("sent_price")),
                        db_text(target.get("sent_error")),
                        db_jsonb(target),
                    ],
                )
        conn.commit()


def record_sync_history(
    paths: dict[str, Path],
    config: SyncConfig,
    result: dict[str, Any],
    *,
    append_legacy_jsonl: bool,
) -> dict[str, Any]:
    payload = {**result, "recorded_at": result.get("recorded_at") or now_iso()}
    try:
        persist_rebalance_result_to_db(config, payload)
    except Exception as exc:
        print(f"failed to persist paper rebalance history to DB: {exc}", file=sys.stderr, flush=True)
    if append_legacy_jsonl:
        append_jsonl(paths["history"], payload)
    return payload


def execute_plan(
    client: GatewayClient,
    config: SyncConfig,
    *,
    plan: pd.DataFrame,
    signal_date: str,
    active_orders: list[dict[str, Any]],
) -> dict[str, Any]:
    cancelled_orders: list[dict[str, Any]] = []
    placed_orders: list[dict[str, Any]] = []
    skipped_orders: list[dict[str, Any]] = []

    execution_rows = plan.copy()
    execution_rows["sent_order_id"] = None
    execution_rows["sent_status"] = None
    execution_rows["sent_price"] = None
    execution_rows["sent_reference_price"] = None
    execution_rows["sent_error"] = None

    if not is_active_trading_hours():
        message = "Outside active trading hours, skipping order execution..."
        print(message, flush=True)
        return {
            "execution_rows": execution_rows,
            "cancelled_orders": cancelled_orders,
            "placed_orders": placed_orders,
            "skipped_orders": skipped_orders,
            "execution_skipped": True,
            "skip_reason": "outside_active_trading_hours",
            "message": message,
        }

    if config.cancel_open_orders:
        for order in active_orders:
            order_id = str(order.get("broker_order_id") or order.get("order_id") or "").strip()
            if not order_id:
                continue
            cancelled_orders.append(client.cancel_order(order_id))

    sell_rows = execution_rows[execution_rows["sell_order_qty"] > 0].sort_values(["score", "rank"], ascending=[True, True], na_position="last")
    buy_rows = execution_rows[execution_rows["buy_order_qty"] > 0].sort_values(["rank", "score"], ascending=[True, False], na_position="last")
    last_order_attempt_at = 0.0

    for side, rows, qty_col in [
        ("SELL", sell_rows, "sell_order_qty"),
        ("BUY", buy_rows, "buy_order_qty"),
    ]:
        for index, row in rows.iterrows():
            quantity = int(row[qty_col])
            if quantity <= 0:
                continue
            max_order_qty = config.max_sell_order_qty if side == "SELL" else config.max_buy_order_qty
            quantity = apply_optional_quantity_cap(quantity, max_order_qty)
            if quantity <= 0:
                continue
            symbol = str(row["code"])
            try:
                latest_price = get_sina_latest_price(symbol, row.get("exchange"))
                price = build_marketable_limit_price(latest_price, side, config.execution_model)
            except GatewayError as exc:
                last_error = str(exc)
                execution_rows.at[index, "sent_status"] = "SKIPPED_NO_LIVE_PRICE"
                execution_rows.at[index, "sent_error"] = last_error
                execution_rows.at[index, "action"] = "SKIP_NO_LIVE_PRICE"
                skipped_orders.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "attempted_prices": [],
                        "error": last_error,
                    }
                )
                print(f"skip {side} {symbol} qty={quantity}: {last_error}", file=sys.stderr, flush=True)
                continue
            if price <= 0:
                execution_rows.at[index, "sent_status"] = "SKIPPED_NO_REFERENCE_PRICE"
                execution_rows.at[index, "action"] = "SKIP_NO_REFERENCE_PRICE"
                skipped_orders.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "attempted_prices": [],
                        "error": "no live or planned reference price available",
                    }
                )
                continue
            reference_previous_close = to_float(row.get("previous_close"))
            if near_price_limit(
                side=side,
                price=latest_price,
                previous_close=reference_previous_close,
                symbol=symbol,
                name=row.get("name"),
                model=config.execution_model,
            ):
                skip_reason = "SKIP_LIMIT_UP" if side == "BUY" else "SKIP_LIMIT_DOWN"
                execution_rows.at[index, "sent_status"] = skip_reason
                execution_rows.at[index, "sent_reference_price"] = round_price(latest_price)
                execution_rows.at[index, "sent_price"] = price
                execution_rows.at[index, "action"] = skip_reason
                skipped_orders.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "attempted_prices": [price],
                        "error": skip_reason,
                    }
                )
                continue
            if side == "BUY":
                skip_reason = buy_liquidity_skip_reason(
                    amount=row.get("amount"),
                    order_notional=quantity * price,
                    model=config.execution_model,
                )
                if skip_reason is not None:
                    execution_rows.at[index, "sent_status"] = skip_reason
                    execution_rows.at[index, "sent_reference_price"] = round_price(latest_price)
                    execution_rows.at[index, "sent_price"] = price
                    execution_rows.at[index, "action"] = skip_reason
                    skipped_orders.append(
                        {
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "attempted_prices": [price],
                            "error": skip_reason,
                        }
                    )
                    continue
            execution_rows.at[index, "sent_reference_price"] = round_price(latest_price)
            remark = f"aistock sig={signal_date} r={row['rank'] or '-'}"
            last_error: str | None = None
            while True:
                try:
                    elapsed = time.monotonic() - last_order_attempt_at
                    if elapsed < ORDER_ATTEMPT_INTERVAL_SECONDS:
                        time.sleep(ORDER_ATTEMPT_INTERVAL_SECONDS - elapsed)
                    last_order_attempt_at = time.monotonic()
                    order = client.place_order(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        remark=remark,
                    )
                    execution_rows.at[index, "sent_order_id"] = str(order.get("order_id") or order.get("broker_order_id") or "")
                    execution_rows.at[index, "sent_status"] = str(order.get("order_status") or "")
                    execution_rows.at[index, "sent_price"] = price
                    placed_orders.append(order)
                    last_error = None
                    break
                except GatewayError as exc:
                    last_error = str(exc)
                    if is_order_rate_limit_error(last_error):
                        print(
                            f"broker rate limit while placing {side} {symbol}; waiting before retry",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(30)
                        continue
                    break

            if last_error:
                execution_rows.at[index, "sent_status"] = "SKIPPED_PRICE_LIMIT" if is_price_limit_error(last_error) else "SKIPPED_ORDER_ERROR"
                execution_rows.at[index, "sent_error"] = last_error
                execution_rows.at[index, "action"] = "SKIP_PRICE_LIMIT" if is_price_limit_error(last_error) else "SKIP_ORDER_ERROR"
                skipped_orders.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "quantity": quantity,
                        "attempted_prices": [price],
                        "error": last_error,
                    }
                )
                print(
                    f"skip {side} {symbol} qty={quantity} at marketable price {price}: {last_error}",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "execution_rows": execution_rows,
        "cancelled_orders": cancelled_orders,
        "placed_orders": placed_orders,
        "skipped_orders": skipped_orders,
        "execution_skipped": False,
    }


def sync_once(config: SyncConfig) -> tuple[int, dict[str, Any]]:
    paths = ensure_dirs(config.state_dir)
    state = read_json(paths["state"])
    last_signal_date = None
    try:
        config, model_context = resolve_paper_model(config)
        state = update_state(paths, active_model=model_context)
        gateway = GatewayClient(config)
        latest_scores, signal_date, signature = load_latest_scores(config)
        last_signal_date = signal_date
        previous_signature = str(state.get("last_score_signature") or "")

        health_payload = gateway.health()
        health_ok = str(health_payload.get("status") or "").lower() == "ok"

        if config.sync_existing_orders:
            try:
                gateway.sync_agent()
            except GatewayError:
                # The sync endpoint is helpful but not critical enough to block order planning.
                pass

        positions = normalize_positions(gateway.get_agent_positions())
        managed_positions = refresh_managed_positions_from_db(config)
        orders = normalize_orders(gateway.get_agent_orders())
        active_orders = [row for row in orders if is_active_order(row.get("order_status"))]
        balance_rows = gateway.get_balance()
        balance_metrics = extract_balance_metrics(balance_rows, config.account_id)
        live_summary = gateway.get_agent_summary()

        plan, plan_summary = build_plan(
            config,
            latest_scores=latest_scores,
            positions=positions,
            balance_metrics=balance_metrics,
            managed_positions=managed_positions,
        )
        rebalance = rebalance_decision(
            scores_path=config.scores_path,
            state=state,
            signal_date=signal_date,
            force=config.force,
        )
        plan_summary = {**plan_summary, **rebalance, "active_model": model_context}
        planned_target_snapshot = target_snapshot_rows(plan)

        if not config.force and previous_signature == signature:
            has_pending_actions = bool(plan_summary.get("buy_order_count")) or bool(plan_summary.get("sell_order_count"))
            noop_message: str | None
            if active_orders:
                noop_message = f"score snapshot {signal_date} is unchanged and {len(active_orders)} active orders are still working"
            elif has_pending_actions:
                noop_message = (
                    f"score snapshot {signal_date} has already been attempted; "
                    "waiting for the next score snapshot before placing more orders"
                )
            elif not has_pending_actions:
                noop_message = f"score snapshot {signal_date} is unchanged and portfolio is already aligned"
            else:
                noop_message = None

            if noop_message is not None:
                safe_write_daily_snapshot(
                    paths,
                    balance_metrics=balance_metrics,
                    live_summary=live_summary,
                    positions=list(positions.values()),
                    orders=orders,
                )
                summary = {
                    "status": "noop",
                    "message": noop_message,
                    "score_signal_date": signal_date,
                    "last_score_signature": signature,
                    "gateway_healthy": health_ok,
                    "balance_metrics": balance_metrics,
                    "plan_summary": plan_summary,
                    "live_summary": live_summary,
                    "active_order_count": len(active_orders),
                    "position_count": len(positions),
                    "cancelled_order_ids": [],
                    "placed_order_ids": [],
                    "skipped_symbols": [],
                    "cancelled_orders": [],
                    "placed_orders": [],
                    "skipped_orders": [],
                    "target_snapshot": planned_target_snapshot,
                }
                updated = update_state(
                    paths,
                    strategy="futu_gateway_auto_paper_trading",
                    market=config.market,
                    agent_id=config.agent_id,
                    gateway_base_url=config.gateway_base_url,
                    config_snapshot={
                        "top_k": config.top_k,
                        "min_score": config.min_score,
                        "lot_size": config.lot_size,
                        "cash_buffer_pct": config.cash_buffer_pct,
                        "budget_total": config.budget_total,
                        "max_buy_order_qty": config.max_buy_order_qty,
                        "max_sell_order_qty": config.max_sell_order_qty,
                        "execution_model": execution_model_snapshot(config.execution_model),
                    },
                    last_attempt_at=now_iso(),
                    last_status="noop",
                    last_message=summary["message"],
                    score_signal_date=signal_date,
                    last_score_signature=signature,
                    last_error=None,
                    last_traceback=None,
                    gateway_healthy=health_ok,
                    live_summary=live_summary,
                    balance_metrics=balance_metrics,
                    plan_summary=plan_summary,
                    rebalance_profile_name=rebalance["profile_name"],
                    rebalance_every=rebalance["rebalance_every"],
                    rebalance_due=rebalance["rebalance_due"],
                    rebalance_wait_count=rebalance["rebalance_wait_count"],
                    rebalance_observed_signal_dates=rebalance["rebalance_observed_signal_dates"],
                    active_order_count=len(active_orders),
                    position_count=len(positions),
                    cancelled_order_ids=[],
                    placed_order_ids=[],
                    skipped_symbols=[],
                    target_snapshot=planned_target_snapshot,
                )
                record_sync_history(paths, config, summary, append_legacy_jsonl=False)
                return 0, updated

        if not rebalance["rebalance_due"]:
            safe_write_daily_snapshot(
                paths,
                balance_metrics=balance_metrics,
                live_summary=live_summary,
                positions=list(positions.values()),
                orders=orders,
            )
            message = (
                f"score snapshot {signal_date} is not due for "
                f"{rebalance['profile_label']} rebalance every {rebalance['rebalance_every']} trading score dates"
            )
            result = {
                "status": "noop",
                "message": message,
                "score_signal_date": signal_date,
                "last_score_signature": signature,
                "gateway_healthy": health_ok,
                "balance_metrics": balance_metrics,
                "plan_summary": plan_summary,
                "live_summary": live_summary,
                "active_order_count": len(active_orders),
                "position_count": len(positions),
                "cancelled_order_ids": [],
                "placed_order_ids": [],
                "skipped_symbols": [],
                "cancelled_orders": [],
                "placed_orders": [],
                "skipped_orders": [],
                "target_snapshot": planned_target_snapshot,
            }
            updated = update_state(
                paths,
                strategy="futu_gateway_auto_paper_trading",
                market=config.market,
                agent_id=config.agent_id,
                gateway_base_url=config.gateway_base_url,
                config_snapshot={
                    "top_k": config.top_k,
                    "min_score": config.min_score,
                    "lot_size": config.lot_size,
                    "cash_buffer_pct": config.cash_buffer_pct,
                    "budget_total": config.budget_total,
                    "max_buy_order_qty": config.max_buy_order_qty,
                    "max_sell_order_qty": config.max_sell_order_qty,
                    "execution_model": execution_model_snapshot(config.execution_model),
                },
                last_attempt_at=now_iso(),
                last_status="noop",
                last_message=message,
                score_signal_date=signal_date,
                last_score_signature=signature,
                last_error=None,
                last_traceback=None,
                gateway_healthy=health_ok,
                live_summary=live_summary,
                balance_metrics=balance_metrics,
                plan_summary=plan_summary,
                rebalance_profile_name=rebalance["profile_name"],
                rebalance_every=rebalance["rebalance_every"],
                rebalance_due=False,
                rebalance_wait_count=rebalance["rebalance_wait_count"],
                rebalance_observed_signal_dates=rebalance["rebalance_observed_signal_dates"],
                active_order_count=len(active_orders),
                position_count=len(positions),
                cancelled_order_ids=[],
                placed_order_ids=[],
                skipped_symbols=[],
                target_snapshot=planned_target_snapshot,
            )
            record_sync_history(paths, config, result, append_legacy_jsonl=False)
            return 0, updated

        if config.dry_run:
            persist_targets(paths, plan)
            safe_write_daily_snapshot(
                paths,
                balance_metrics=balance_metrics,
                live_summary=live_summary,
                positions=list(positions.values()),
                orders=orders,
            )
            result = {
                "status": "dry_run",
                "message": f"dry run built a rebalance plan for {signal_date}",
                "score_signal_date": signal_date,
                "last_score_signature": signature,
                "gateway_healthy": health_ok,
                "balance_metrics": balance_metrics,
                "plan_summary": plan_summary,
                "live_summary": live_summary,
                "active_order_count": len(active_orders),
                "position_count": len(positions),
                "cancelled_order_ids": [],
                "placed_order_ids": [],
                "skipped_symbols": [],
                "cancelled_orders": [],
                "placed_orders": [],
                "skipped_orders": [],
                "target_snapshot": planned_target_snapshot,
            }
            updated = update_state(
                paths,
                strategy="futu_gateway_auto_paper_trading",
                market=config.market,
                agent_id=config.agent_id,
                gateway_base_url=config.gateway_base_url,
                config_snapshot={
                    "top_k": config.top_k,
                    "min_score": config.min_score,
                    "lot_size": config.lot_size,
                    "cash_buffer_pct": config.cash_buffer_pct,
                    "budget_total": config.budget_total,
                    "max_buy_order_qty": config.max_buy_order_qty,
                    "max_sell_order_qty": config.max_sell_order_qty,
                    "execution_model": execution_model_snapshot(config.execution_model),
                },
                last_attempt_at=now_iso(),
                last_status="dry_run",
                last_message=result["message"],
                score_signal_date=signal_date,
                last_score_signature=signature,
                last_error=None,
                last_traceback=None,
                last_success_at=now_iso(),
                gateway_healthy=health_ok,
                live_summary=live_summary,
                balance_metrics=balance_metrics,
                plan_summary=plan_summary,
                rebalance_profile_name=rebalance["profile_name"],
                rebalance_every=rebalance["rebalance_every"],
                rebalance_due=rebalance["rebalance_due"],
                rebalance_wait_count=rebalance["rebalance_wait_count"],
                rebalance_observed_signal_dates=rebalance["rebalance_observed_signal_dates"],
                active_order_count=len(active_orders),
                position_count=len(positions),
                target_snapshot=planned_target_snapshot,
            )
            record_sync_history(paths, config, result, append_legacy_jsonl=False)
            return 0, updated

        execution = execute_plan(
            gateway,
            config,
            plan=plan,
            signal_date=signal_date,
            active_orders=active_orders,
        )
        execution_rows = execution["execution_rows"]
        persist_targets(paths, execution_rows)
        executed_target_snapshot = target_snapshot_rows(execution_rows)

        if execution.get("execution_skipped"):
            safe_write_daily_snapshot(
                paths,
                balance_metrics=balance_metrics,
                live_summary=live_summary,
                positions=list(positions.values()),
                orders=orders,
            )
            result = {
                "status": "noop",
                "message": str(execution.get("message") or "order execution skipped"),
                "score_signal_date": signal_date,
                "last_score_signature": signature,
                "gateway_healthy": health_ok,
                "balance_metrics": balance_metrics,
                "plan_summary": {**plan_summary, "execution_skip_reason": execution.get("skip_reason")},
                "live_summary": live_summary,
                "active_order_count": len(active_orders),
                "position_count": len(positions),
                "cancelled_order_ids": [],
                "placed_order_ids": [],
                "skipped_symbols": [],
                "cancelled_orders": [],
                "placed_orders": [],
                "skipped_orders": [],
                "target_snapshot": executed_target_snapshot,
            }
            updated = update_state(
                paths,
                strategy="futu_gateway_auto_paper_trading",
                market=config.market,
                agent_id=config.agent_id,
                gateway_base_url=config.gateway_base_url,
                config_snapshot={
                    "top_k": config.top_k,
                    "min_score": config.min_score,
                    "lot_size": config.lot_size,
                    "cash_buffer_pct": config.cash_buffer_pct,
                    "budget_total": config.budget_total,
                    "max_buy_order_qty": config.max_buy_order_qty,
                    "max_sell_order_qty": config.max_sell_order_qty,
                    "execution_model": execution_model_snapshot(config.execution_model),
                },
                last_attempt_at=now_iso(),
                last_status="noop",
                last_message=result["message"],
                score_signal_date=signal_date,
                last_error=None,
                last_traceback=None,
                gateway_healthy=health_ok,
                live_summary=live_summary,
                balance_metrics=balance_metrics,
                plan_summary=result["plan_summary"],
                rebalance_profile_name=rebalance["profile_name"],
                rebalance_every=rebalance["rebalance_every"],
                rebalance_due=rebalance["rebalance_due"],
                rebalance_wait_count=rebalance["rebalance_wait_count"],
                rebalance_observed_signal_dates=rebalance["rebalance_observed_signal_dates"],
                active_order_count=len(active_orders),
                position_count=len(positions),
                cancelled_order_ids=[],
                placed_order_ids=[],
                skipped_symbols=[],
                target_snapshot=executed_target_snapshot,
            )
            record_sync_history(paths, config, result, append_legacy_jsonl=True)
            return 0, updated

        try:
            gateway.sync_agent()
        except GatewayError:
            pass

        live_summary = gateway.get_agent_summary()
        balance_rows = gateway.get_balance()
        balance_metrics = extract_balance_metrics(balance_rows, config.account_id)
        refreshed_positions = normalize_positions(gateway.get_agent_positions())
        refresh_managed_positions_from_db(config)
        refreshed_orders = normalize_orders(gateway.get_agent_orders())
        safe_write_daily_snapshot(
            paths,
            balance_metrics=balance_metrics,
            live_summary=live_summary,
            positions=list(refreshed_positions.values()),
            orders=refreshed_orders,
        )
        execution_skip_count = len(execution.get("skipped_orders", []))
        plan_summary = {**plan_summary, "execution_skip_count": execution_skip_count}
        skipped_symbols = [str(item.get("symbol") or "") for item in execution.get("skipped_orders", []) if item]
        message = (
            f"rebalance synced for {signal_date}: "
            f"{len(execution['cancelled_orders'])} cancellations, "
            f"{len(execution['placed_orders'])} new orders"
        )
        if skipped_symbols:
            message += f", skipped {len(skipped_symbols)} symbols ({', '.join(skipped_symbols)})"
        result = {
            "status": "success",
            "message": message,
            "score_signal_date": signal_date,
            "last_score_signature": signature,
            "gateway_healthy": health_ok,
            "balance_metrics": balance_metrics,
            "plan_summary": plan_summary,
            "live_summary": live_summary,
            "active_order_count": len([row for row in refreshed_orders if is_active_order(row.get('order_status'))]),
            "position_count": len(positions),
            "cancelled_order_ids": [
                str(item.get("order_id") or item.get("broker_order_id") or "")
                for item in execution["cancelled_orders"]
                if item
            ],
            "placed_order_ids": [
                str(item.get("order_id") or item.get("broker_order_id") or "")
                for item in execution["placed_orders"]
                if item
            ],
            "skipped_symbols": skipped_symbols,
            "cancelled_orders": [snapshot_order_event(item) for item in execution["cancelled_orders"] if item],
            "placed_orders": [snapshot_order_event(item) for item in execution["placed_orders"] if item],
            "skipped_orders": [snapshot_skipped_order(item) for item in execution["skipped_orders"] if item],
            "target_snapshot": executed_target_snapshot,
        }
        updated = update_state(
            paths,
            strategy="futu_gateway_auto_paper_trading",
            market=config.market,
            agent_id=config.agent_id,
            gateway_base_url=config.gateway_base_url,
            config_snapshot={
                "top_k": config.top_k,
                "min_score": config.min_score,
                "lot_size": config.lot_size,
                "cash_buffer_pct": config.cash_buffer_pct,
                "budget_total": config.budget_total,
                "max_buy_order_qty": config.max_buy_order_qty,
                "max_sell_order_qty": config.max_sell_order_qty,
                "execution_model": execution_model_snapshot(config.execution_model),
            },
            last_attempt_at=now_iso(),
            last_success_at=now_iso(),
            last_status="success",
            last_message=result["message"],
            score_signal_date=signal_date,
            last_applied_signal_date=signal_date,
            last_score_signature=signature,
            last_error=None,
            last_traceback=None,
            gateway_healthy=health_ok,
            live_summary=live_summary,
            balance_metrics=balance_metrics,
            plan_summary=plan_summary,
            rebalance_profile_name=rebalance["profile_name"],
            rebalance_every=rebalance["rebalance_every"],
            rebalance_due=rebalance["rebalance_due"],
            rebalance_wait_count=rebalance["rebalance_wait_count"],
            rebalance_observed_signal_dates=[],
            active_order_count=result["active_order_count"],
            position_count=len(positions),
            cancelled_order_ids=result["cancelled_order_ids"],
            placed_order_ids=result["placed_order_ids"],
            skipped_symbols=result["skipped_symbols"],
            target_snapshot=executed_target_snapshot,
        )
        record_sync_history(paths, config, result, append_legacy_jsonl=True)
        return 0, updated
    except Exception as exc:
        message = str(exc)
        repeated_error = state.get("last_status") == "error" and state.get("last_error") == message
        repeat_count = int(state.get("last_error_repeat_count") or 0) + 1 if repeated_error else 1
        failure = update_state(
            paths,
            strategy="futu_gateway_auto_paper_trading",
            market=config.market,
            agent_id=config.agent_id,
            gateway_base_url=config.gateway_base_url,
            last_attempt_at=now_iso(),
            last_status="error",
            last_message=message,
            score_signal_date=last_signal_date,
            last_error=message,
            last_error_repeat_count=repeat_count,
            last_error_first_at=(state.get("last_error_first_at") if repeated_error else now_iso()),
            last_traceback=traceback.format_exc(limit=20),
        )
        if not repeated_error:
            record_sync_history(
                paths,
                config,
                {
                    "status": "error",
                    "message": message,
                    "score_signal_date": last_signal_date,
                },
                append_legacy_jsonl=True,
            )
        print(message, file=sys.stderr, flush=True)
        return 1, failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto paper-trading reconciler for the Futu gateway.")
    parser.add_argument("--scores-path", default=DEFAULT_SCORES_PATH)
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--gateway-base-url", default=DEFAULT_GATEWAY_BASE_URL)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    parser.add_argument("--agent-key", default=DEFAULT_AGENT_KEY)
    parser.add_argument("--agent-id-header", default=DEFAULT_AGENT_ID_HEADER)
    parser.add_argument("--agent-key-header", default=DEFAULT_AGENT_KEY_HEADER)
    parser.add_argument("--account-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--lot-size", type=int, default=DEFAULT_LOT_SIZE)
    parser.add_argument("--cash-buffer-pct", type=float, default=DEFAULT_CASH_BUFFER_PCT)
    parser.add_argument("--buy-limit-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sell-limit-bps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--budget-total", type=float, default=None)
    parser.add_argument("--max-order-qty", type=int, default=DEFAULT_MAX_ORDER_QTY)
    parser.add_argument("--max-buy-order-qty", type=int, default=None)
    parser.add_argument("--max-sell-order-qty", type=int, default=None)
    parser.add_argument("--paper-db-url", default=os.getenv("PAPER_DB_URL"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cancel-open-orders", action="store_true")
    parser.add_argument("--no-sync-existing-orders", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SyncConfig:
    execution_model = execution_model_with_limit_bps(
        buy_limit_bps=args.buy_limit_bps,
        sell_limit_bps=args.sell_limit_bps,
    )
    return SyncConfig(
        scores_path=Path(args.scores_path),
        state_dir=Path(args.state_dir),
        gateway_base_url=str(args.gateway_base_url).strip() or DEFAULT_GATEWAY_BASE_URL,
        market=str(args.market).strip().upper() or DEFAULT_MARKET,
        agent_id=str(args.agent_id).strip() or DEFAULT_AGENT_ID,
        agent_key=str(args.agent_key).strip() or DEFAULT_AGENT_KEY,
        agent_id_header=str(args.agent_id_header).strip() or DEFAULT_AGENT_ID_HEADER,
        agent_key_header=str(args.agent_key_header).strip() or DEFAULT_AGENT_KEY_HEADER,
        account_id=args.account_id,
        top_k=max(int(args.top_k), 1),
        min_score=float(args.min_score),
        lot_size=max(int(args.lot_size), 1),
        cash_buffer_pct=max(min(float(args.cash_buffer_pct), 0.95), 0.0),
        budget_total=float(args.budget_total) if args.budget_total is not None else None,
        max_buy_order_qty=max(int(args.max_buy_order_qty if args.max_buy_order_qty is not None else args.max_order_qty), 0),
        max_sell_order_qty=max(int(args.max_sell_order_qty if args.max_sell_order_qty is not None else args.max_order_qty), 1),
        cancel_open_orders=not bool(args.no_cancel_open_orders),
        sync_existing_orders=not bool(args.no_sync_existing_orders),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        paper_db_url=(str(getattr(args, "paper_db_url", None) or os.getenv("PAPER_DB_URL") or "").strip() or None),
        execution_model=execution_model,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)
    code, state = sync_once(config)
    print(json.dumps(state, ensure_ascii=False, indent=2, default=json_default))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
