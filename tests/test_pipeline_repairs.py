from __future__ import annotations

import sys
import json
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "api"))

import batch_download_all_a as batch_download
from batch_download_all_a import mark_existing_outputs, merge_existing_output, process_code
from backtest_walk_forward import annualized_return, estimate_rebalance_fees, max_drawdown, training_end_for_rebalance
from build_inference_features import build_inference_frame
from control_settings import is_investable_stock_name, is_st_stock_name, read_control_settings, write_control_settings
from execution_model import (
    DEFAULT_EXECUTION_MODEL,
    board_price_limit_rate,
    buy_liquidity_skip_reason,
    liquidity_cap_notional,
    near_price_limit,
)
from download_data import (
    build_valuation_df,
    is_investable_stock_name as is_universe_investable_stock_name,
    reference_status_path,
    write_reference_status,
)
from repair_valuation_reference_fields import repair_one
from paper_trade_futu import (
    SyncConfig,
    build_plan,
    compute_affordable_buy_quantity,
    execute_plan,
    load_latest_scores,
    parse_sina_quote_price,
    persist_targets,
    sina_quote_code,
    score_file_signature,
    sync_once,
    write_daily_snapshot,
)
from trading_fees import DEFAULT_FEE_MODEL, transaction_fee
from scripts.import_daily_kline_to_postgres import UPSERT_SQL as KLINE_UPSERT_SQL, rows_from_kline
from scripts.import_stock_list_to_postgres import UPSERT_SQL as STOCK_LIST_UPSERT_SQL
from scripts.update_us_selection_data import process_symbol_batches, shares_to_yi
from scripts.import_stock_master_attributes import (
    SHAREHOLDER_RESEARCH_UPSERT_SQL,
    eastmoney_secucode,
    parse_shareholder_research_rows,
    parse_sina_concept_html,
)
import refresh_reference_data as reference_refresh
from app.services import batch as batch_service
from app.services import admin_settings as admin_settings_service
from app.services import benchmark as benchmark_service
from app.services import model as model_service
from app.services import model_profiles as model_profiles_service
from app.services import paper as paper_service
from app.services import paper_db as paper_db_service
from app.services import pipeline_control as pipeline_control_service
from app.services import pipeline_runner as pipeline_runner_service
from app.services import reference_control as reference_control_service
from app.services import source_readiness
from app.services import us_selection_control as us_selection_control_service
from app.services.fei_selection import SELECTION_SQL, STOCK_DETAIL_HISTORY_SQL, STOCK_DETAIL_SHAREHOLDER_SQL
from app.services.log_translation import translate_log_line
from app.services.us_selection_control import _parse_time


class PipelineRepairTests(unittest.TestCase):
    @staticmethod
    def _mock_model_registry(settings: SimpleNamespace):
        metadata_path = settings.models_dir / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        active_profile = str(metadata.get("profile_name") or "short_5d")
        deployment = {
            "market": "CN",
            "model_id": "active-model",
            "model_version": f"cn-{active_profile}-test",
            "profile": active_profile,
            "artifact_path": str(settings.models_dir),
            "artifact_manifest": {},
            "paper_enabled": True,
            "validation_status": "legacy_unreviewed",
            "revision": 1,
        }

        def latest(_market: str, profile: str, **_kwargs: object) -> dict[str, object] | None:
            artifact_dir = settings.quant_dir / "model_profiles" / profile / "models"
            if not artifact_dir.exists():
                return None
            return {**deployment, "model_id": f"{profile}-model", "model_version": f"cn-{profile}-test", "profile": profile, "artifact_path": str(artifact_dir)}

        return mock.patch.multiple(
            model_service,
            get_active_deployment=mock.Mock(return_value=deployment),
            get_latest_model_for_profile=mock.Mock(side_effect=latest),
            list_model_versions=mock.Mock(return_value=[deployment]),
            list_activation_events=mock.Mock(return_value=[]),
            sync_model_registry=mock.Mock(return_value={"registered": 0, "deployment": deployment}),
            resolve_artifact_path=mock.Mock(side_effect=lambda value, _settings: Path(str(value))),
        )

    @staticmethod
    def _paper_sync_config(scores_path: Path, state_dir: Path, *, force: bool = False) -> SyncConfig:
        return SyncConfig(
            scores_path=scores_path,
            state_dir=state_dir,
            gateway_base_url="http://127.0.0.1:8080",
            market="CN",
            agent_id="agent",
            agent_key="key",
            agent_id_header="X-Agent-Id",
            agent_key_header="X-Agent-Key",
            account_id=None,
            top_k=1,
            min_score=0.5,
            lot_size=100,
            cash_buffer_pct=0.0,
            budget_total=None,
            max_buy_order_qty=1000,
            max_sell_order_qty=1000,
            cancel_open_orders=True,
            sync_existing_orders=True,
            force=force,
            dry_run=False,
        )

    @staticmethod
    def _write_scores(path: Path, dates: list[str], *, profile_name: str = "short_5d", label_horizon: int = 5) -> None:
        pd.DataFrame(
            [
                {
                    "date": trade_date,
                    "code": "000001",
                    "exchange": "SZ",
                    "name": "Ping An Bank",
                    "industry": "Bank",
                    "score": 0.9,
                    "close": 10.0,
                }
                for trade_date in dates
            ]
        ).to_parquet(path, index=False)
        (path.parent / "training_metadata.json").write_text(
            json.dumps({"profile_name": profile_name, "label_horizon": label_horizon}),
            encoding="utf-8",
        )

    @staticmethod
    def _pending_plan() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-20",
                    "rank": 1,
                    "code": "000001",
                    "score": 0.9,
                    "target_qty": 100,
                    "current_qty": 0,
                    "delta_qty": 100,
                    "sell_order_qty": 0,
                    "buy_order_qty": 100,
                    "estimated_order_notional": 1000.0,
                    "estimated_order_fee": 1.0,
                    "current_market_value": 0.0,
                    "action": "BUY",
                }
            ]
        )

    @staticmethod
    def _noop_gateway_class():
        class NoopGateway:
            def __init__(self, config: SyncConfig) -> None:
                self.config = config

            def health(self) -> dict[str, object]:
                return {"status": "ok"}

            def sync_agent(self) -> None:
                return None

            def get_agent_positions(self) -> list[dict[str, object]]:
                return []

            def get_agent_orders(self) -> list[dict[str, object]]:
                return []

            def get_balance(self) -> list[dict[str, object]]:
                return [{"cash": 1000.0, "power": 1000.0, "total_assets": 1000.0}]

            def get_agent_summary(self) -> dict[str, object]:
                return {"total_assets": 1000.0, "total_pnl": 0.0}

        return NoopGateway

    def test_sina_concept_parser_extracts_industry_and_concepts(self) -> None:
        payload = """
        <html><body>
          <table class="comInfo1">
            <tr><td class="ct" colspan="2">所属行业板块</td></tr>
            <tr><td>板块名称</td><td>同行业个股</td></tr>
            <tr><td>软件服务</td><td>demo</td></tr>
            <tr><td>备注：此为申万行业分类</td></tr>
          </table>
          <table class="comInfo1">
            <tr><td class="ct" colspan="2">所属概念板块</td></tr>
            <tr><td>板块名称</td><td>相关个股</td></tr>
            <tr><td><a>人工智能</a></td><td>demo</td></tr>
            <tr><td>国产软件</td><td>demo</td></tr>
          </table>
        </body></html>
        """

        result = parse_sina_concept_html(payload)

        self.assertFalse(result.blocked)
        self.assertEqual(result.industry, ["软件服务"])
        self.assertEqual(result.concepts, ["人工智能", "国产软件"])

    def test_eastmoney_shareholder_secucode_format(self) -> None:
        self.assertEqual(eastmoney_secucode("600584", "sh"), "600584.SH")
        self.assertEqual(eastmoney_secucode("000001", "sz"), "000001.SZ")

    def test_eastmoney_shareholder_rows_parse_fixture_and_cutoff(self) -> None:
        rows = parse_shareholder_research_rows(
            [
                {
                    "SECUCODE": "600584.SH",
                    "SECURITY_CODE": "600584",
                    "END_DATE": "2026-03-31 00:00:00",
                    "HOLDER_TOTAL_NUM": 320364,
                    "TOTAL_NUM_RATIO": -12.6652,
                    "AVG_FREE_SHARES": 5585,
                    "AVG_FREESHARES_RATIO": 14.501941541497,
                    "HOLD_FOCUS": "非常分散",
                },
                {
                    "SECUCODE": "600584.SH",
                    "SECURITY_CODE": "600584",
                    "END_DATE": "2023-12-31 00:00:00",
                    "HOLDER_TOTAL_NUM": 230100,
                    "TOTAL_NUM_RATIO": -1,
                    "AVG_FREE_SHARES": 7775,
                    "AVG_FREESHARES_RATIO": 1,
                    "HOLD_FOCUS": "非常分散",
                },
                {
                    "SECUCODE": "600584.SH",
                    "SECURITY_CODE": "600584",
                    "END_DATE": "2024-06-30 00:00:00",
                    "HOLDER_TOTAL_NUM": 230100,
                    "TOTAL_NUM_RATIO": -19.63,
                    "AVG_FREE_SHARES": 7775,
                    "AVG_FREESHARES_RATIO": 24.45,
                    "HOLD_FOCUS": "非常分散",
                },
            ],
            code="600584",
            exchange="sh",
            secucode="600584.SH",
            start_date=date(2024, 1, 1),
        )

        self.assertEqual([row.report_date for row in rows], [date(2026, 3, 31), date(2024, 6, 30)])
        self.assertEqual(rows[0].holder_total_num, 320364)
        self.assertEqual(rows[0].total_num_ratio, -12.6652)
        self.assertEqual(rows[0].avg_free_shares, 5585)
        self.assertEqual(rows[0].avg_freeshares_ratio, 14.501941541497)
        self.assertEqual(rows[0].hold_focus, "非常分散")

    def test_shareholder_research_schema_and_upsert_are_idempotent(self) -> None:
        schema = (ROOT / "scripts" / "create_stock_master.sql").read_text(encoding="utf-8")

        self.assertIn("create table if not exists stock_shareholder_research", schema)
        self.assertIn("primary key (report_date, code, exchange)", schema)
        self.assertIn("stock_shareholder_research_code_date_idx", schema)
        self.assertIn("on conflict (report_date, code, exchange) do update set", SHAREHOLDER_RESEARCH_UPSERT_SQL)

    def test_stock_list_upsert_preserves_existing_industry_when_source_is_missing(self) -> None:
        self.assertIn("industry_code = coalesce(excluded.industry_code, stock_master.industry_code)", STOCK_LIST_UPSERT_SQL)
        self.assertIn("industry_name = coalesce(excluded.industry_name, stock_master.industry_name)", STOCK_LIST_UPSERT_SQL)
        self.assertIn("industry_short_name = coalesce(excluded.industry_short_name, stock_master.industry_short_name)", STOCK_LIST_UPSERT_SQL)
        self.assertNotIn("industry_name = excluded.industry_name", STOCK_LIST_UPSERT_SQL)

    def test_reference_publish_runs_full_fei_stock_attributes(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
            calls.append(command)
            return SimpleNamespace(returncode=0)

        with mock.patch.dict(reference_refresh.os.environ, {"PAPER_DB_URL": "postgresql://example/db"}, clear=True):
            with mock.patch.object(reference_refresh.subprocess, "run", side_effect=fake_run):
                reference_refresh.run_fei_stock_attributes(Path("quant_data"))

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][1].endswith("scripts/import_stock_list_to_postgres.py"))
        self.assertTrue(calls[1][1].endswith("scripts/import_stock_master_attributes.py"))
        self.assertNotIn("--skip-eps", calls[1])
        self.assertIn("run/fei_stock_attributes_status.json", calls[1])
        self.assertIn("run/fei_stock_attributes_checkpoint.json", calls[1])

    def test_pipeline_stock_master_upsert_runs_import_command(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
            calls.append(command)
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        settings = SimpleNamespace(
            project_root=ROOT,
            stock_list_path=ROOT / "quant_data" / "stock_list.parquet",
            paper_db_url="postgresql://example/db",
            run_dir=ROOT / "run",
        )

        with mock.patch.object(pipeline_runner_service, "get_settings", return_value=settings):
            with mock.patch.object(pipeline_runner_service, "_write_state"):
                with mock.patch.object(pipeline_runner_service, "read_json", return_value={}):
                    with mock.patch.object(pipeline_runner_service, "_log"):
                        with mock.patch.object(pipeline_runner_service.subprocess, "run", side_effect=fake_run):
                            pipeline_runner_service._run_stock_master_upsert()

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][1].endswith("scripts/import_stock_list_to_postgres.py"))
        self.assertIn("--database-url", calls[0])

    def test_daily_kline_import_rows_include_volume_and_amount(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "000001.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-21",
                        "code": "000001",
                        "exchange": "sz",
                        "close": 10.5,
                        "volume": 123456,
                        "amount": 6543210,
                        "turnover": 1.23,
                    }
                ]
            ).to_parquet(path, index=False)

            rows = rows_from_kline(path)

        self.assertEqual(rows, [(datetime(2026, 5, 21).date(), "000001", "sz", 10.5, 123456, 6543210, None, 1.23)])

    def test_daily_kline_upsert_preserves_stcn_average_trade(self) -> None:
        self.assertIn(
            "average_trade = coalesce(excluded.average_trade, stock_daily_metrics.average_trade)",
            KLINE_UPSERT_SQL,
        )

    def test_fei_selection_lobster_sql_uses_only_core_lobster_conditions_for_flag(self) -> None:
        self.assertIn("p.volume_3 < p.volume_2", SELECTION_SQL)
        self.assertIn("p.turnover_1 <= 10", SELECTION_SQL)
        self.assertIn("p.close_3d_base", SELECTION_SQL)
        self.assertNotIn("close_lobster_base", SELECTION_SQL)
        self.assertIn("else true", SELECTION_SQL)
        self.assertIn("where lobster_flg", SELECTION_SQL)
        self.assertIn("p.amount_1 / (p.close * sm.float_shares)", SELECTION_SQL)
        self.assertNotIn("is_active", SELECTION_SQL)
        self.assertNotIn("startswith", SELECTION_SQL.lower())

    def test_fei_selection_sql_requires_recent_stcn_and_turnover_coverage(self) -> None:
        self.assertIn("count(average_trade) filter (where rn <= 4) as average_trade_latest4_count", SELECTION_SQL)
        self.assertIn("count(turnover) filter (where rn <= 4) as turnover_latest4_count", SELECTION_SQL)
        self.assertIn("s.average_trade_latest4_count > 0", SELECTION_SQL)
        self.assertIn("s.turnover_latest4_count > 0", SELECTION_SQL)

    def test_fei_selection_sql_exposes_legacy_signal_flags(self) -> None:
        self.assertIn("max(average_trade) filter (where rn = 6) as average_trade_6", SELECTION_SQL)
        self.assertIn("average_trade_over_pct", SELECTION_SQL)
        self.assertIn("turnover_compare_pct", SELECTION_SQL)
        self.assertIn("green_flg", SELECTION_SQL)
        self.assertIn("yellow_flg", SELECTION_SQL)
        self.assertIn("blue_flg", SELECTION_SQL)
        self.assertIn("coalesce(s.average_trade_over_pct >= 30, false) as green_flg", SELECTION_SQL)
        self.assertIn("coalesce(s.turnover_compare_pct >= 250, false) as yellow_flg", SELECTION_SQL)
        self.assertIn("coalesce(s.turnover_compare_pct <= -40, false) as blue_flg", SELECTION_SQL)

    def test_fei_selection_sql_exposes_latest_shareholder_count(self) -> None:
        self.assertIn("shareholder_ranked as", SELECTION_SQL)
        self.assertIn("shareholder_latest as", SELECTION_SQL)
        self.assertIn("shareholder_peak_ranked as", SELECTION_SQL)
        self.assertIn("shareholder_peak as", SELECTION_SQL)
        self.assertIn("from stock_shareholder_research", SELECTION_SQL)
        self.assertIn("max(holder_total_num) filter (where rn = 1) as shareholder_total_num", SELECTION_SQL)
        self.assertIn("holder_total_num) filter (where rn = 2) as shareholder_previous_total_num", SELECTION_SQL)
        self.assertIn("sh.shareholder_total_num - sh.shareholder_previous_total_num", SELECTION_SQL)
        self.assertIn("shareholder_total_num_change_pct", SELECTION_SQL)
        self.assertIn("interval '90 days'", SELECTION_SQL)
        self.assertIn("sp.shareholder_peak_report_date", SELECTION_SQL)
        self.assertIn("sp.shareholder_peak_total_num", SELECTION_SQL)
        self.assertIn("shareholder_total_num_from_peak_change", SELECTION_SQL)
        self.assertIn("shareholder_total_num_from_peak_change_pct", SELECTION_SQL)
        self.assertIn("s.shareholder_total_num", SELECTION_SQL)

    def test_fei_stock_detail_history_sql_includes_volume(self) -> None:
        self.assertIn("volume", STOCK_DETAIL_HISTORY_SQL)
        self.assertIn("average_trade", STOCK_DETAIL_HISTORY_SQL)
        self.assertIn("turnover", STOCK_DETAIL_HISTORY_SQL)

    def test_fei_stock_detail_sql_includes_shareholder_research(self) -> None:
        self.assertIn("from stock_shareholder_research", STOCK_DETAIL_SHAREHOLDER_SQL)
        self.assertIn("holder_total_num", STOCK_DETAIL_SHAREHOLDER_SQL)
        self.assertIn("avg_free_shares", STOCK_DETAIL_SHAREHOLDER_SQL)
        self.assertIn("limit %s", STOCK_DETAIL_SHAREHOLDER_SQL)

    def test_us_selection_schema_has_batch_tables_and_idempotent_guards(self) -> None:
        schema = (ROOT / "scripts" / "create_us_selection.sql").read_text(encoding="utf-8")

        self.assertIn("create table if not exists us_stock_master", schema)
        self.assertIn("create table if not exists us_stock_daily_metrics", schema)
        self.assertIn("create table if not exists us_selection_job_runs", schema)
        self.assertIn("us_selection_job_runs_completed_once_idx", schema)
        self.assertIn("primary key (trade_date, symbol)", schema)

    def test_us_selection_helpers_match_legacy_units_and_schedule_parsing(self) -> None:
        self.assertEqual(shares_to_yi("1.04亿"), 1.04)
        self.assertEqual(shares_to_yi("6755.63万"), 0.675563)
        self.assertEqual(_parse_time("17:10", "00:00"), (17, 10))
        self.assertEqual(_parse_time("bad", "03:00"), (3, 0))

    def test_us_selection_details_scheduler_keeps_catching_up_missing_shares(self) -> None:
        settings = SimpleNamespace(
            run_dir=Path("/tmp"),
            us_selection_price_time="16:31",
            us_selection_average_time="00:30",
            us_selection_details_time="03:00",
            us_selection_universe_time="06:00",
        )
        written_states: list[dict[str, object]] = []

        def fake_start(mode: str, target_date: date | None, state_key: str, state: dict[str, object]) -> dict[str, object]:
            return {
                **state,
                f"last_{state_key}": "2026-05-31",
                "last_triggered_mode": mode,
                "target_date": target_date.isoformat() if target_date else None,
            }

        with (
            mock.patch.object(us_selection_control_service, "get_settings", return_value=settings),
            mock.patch.object(
                us_selection_control_service,
                "_ny_now",
                return_value=datetime(2026, 5, 31, 1, 0, tzinfo=timezone.utc),
            ),
            mock.patch.object(us_selection_control_service, "read_json", return_value={"last_details_local_date": "2026-05-31"}),
            mock.patch.object(us_selection_control_service, "_us_details_missing_count", return_value=12),
            mock.patch.object(us_selection_control_service, "_maybe_start_scheduled_lane", side_effect=fake_start) as start_mock,
            mock.patch.object(us_selection_control_service, "_write_scheduler_state", side_effect=written_states.append),
        ):
            us_selection_control_service._maybe_start_us_selection_jobs()

        start_mock.assert_called_once()
        self.assertEqual(start_mock.call_args.args[:3], ("details", None, "details_local_date"))
        self.assertEqual(written_states[-1]["details_missing_count"], 12)
        self.assertEqual(written_states[-1]["last_triggered_mode"], "details")

    def test_us_selection_details_scheduler_does_not_start_duplicate_details_container(self) -> None:
        running_container = SimpleNamespace(name="aistockcn-us-selection-details-demo", status="running")

        with (
            mock.patch.object(us_selection_control_service, "_find_running_container", return_value=running_container),
            mock.patch.object(us_selection_control_service, "start_us_selection") as start_mock,
        ):
            state = us_selection_control_service._maybe_start_scheduled_lane("details", None, "details_local_date", {})

        start_mock.assert_not_called()
        self.assertEqual(state["last_skip_reason"], "details_already_running")

    def test_us_selection_scheduled_lane_marks_attempt_separately_from_completion(self) -> None:
        with (
            mock.patch.object(us_selection_control_service, "_find_running_container", return_value=None),
            mock.patch.object(us_selection_control_service, "_completed_run_exists", return_value=False),
            mock.patch.object(us_selection_control_service, "start_us_selection", return_value={"code": "started", "container_name": "demo"}),
        ):
            state = us_selection_control_service._maybe_start_scheduled_lane(
                "average-trade",
                date(2026, 7, 10),
                "average_date",
                {},
            )

        self.assertNotIn("last_average_date", state)
        self.assertEqual(state["last_attempted_average_date"], "2026-07-10")

    def test_us_selection_scheduled_lane_marks_date_after_completed_run(self) -> None:
        with (
            mock.patch.object(us_selection_control_service, "_find_running_container", return_value=None),
            mock.patch.object(us_selection_control_service, "_completed_run_exists", return_value=True),
            mock.patch.object(us_selection_control_service, "start_us_selection") as start_mock,
        ):
            state = us_selection_control_service._maybe_start_scheduled_lane(
                "average-trade",
                date(2026, 7, 10),
                "average_date",
                {},
            )

        start_mock.assert_not_called()
        self.assertEqual(state["last_average_date"], "2026-07-10")

    def test_reference_control_passes_symbol_timeout(self) -> None:
        command = reference_control_service._reference_command()

        self.assertIn("--symbol-timeout-seconds", command)
        timeout_index = command.index("--symbol-timeout-seconds") + 1
        self.assertEqual(command[timeout_index], "300")

    def test_us_selection_price_lane_keeps_turnover_dependent_on_existing_shares(self) -> None:
        source = (ROOT / "scripts" / "update_us_selection_data.py").read_text(encoding="utf-8")

        self.assertIn("turnover = None", source)
        self.assertIn("if volume is not None and shares_yi and shares_yi > 0:", source)
        self.assertIn("turnover = round(volume / (shares_yi * 100000000) * 100, 2)", source)

    def test_us_selection_batches_write_requested_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status_path = root / "status.json"
            checkpoint_path = root / "custom-checkpoint.json"

            summary = process_symbol_batches(
                None,
                ["AAPL", "MSFT"],
                status_path,
                None,
                stage="update_prices",
                batch_size=10,
                sleep_seconds=0,
                worker=lambda symbol: True,
                target_date=date(2026, 6, 3),
                checkpoint_path=checkpoint_path,
            )

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["done_count"], 2)
            self.assertEqual(checkpoint["last_symbol"], "MSFT")
            self.assertEqual(checkpoint["target_date"], "2026-06-03")

    def test_paper_db_symbol_ledger_keeps_avg_and_realized_pnl_separate(self) -> None:
        fills = [
            {
                "created_at": datetime(2026, 3, 27, 1, 32),
                "broker_order_id": "buy-1",
                "symbol": "600703",
                "side": "BUY",
                "quantity": 100,
                "price": 12.58,
                "notional": 1258.0,
            },
            {
                "created_at": datetime(2026, 3, 30, 1, 34),
                "broker_order_id": "buy-2",
                "symbol": "600703",
                "side": "BUY",
                "quantity": 100,
                "price": 11.94,
                "notional": 1194.0,
            },
            {
                "created_at": datetime(2026, 5, 21, 1, 37),
                "broker_order_id": "sell-1",
                "symbol": "600703",
                "side": "SELL",
                "quantity": 100,
                "price": 16.72,
                "notional": 1672.0,
            },
        ]

        ledger, daily = paper_db_service.build_symbol_ledger(fills)

        self.assertAlmostEqual(ledger[1]["avg_cost_after"], 12.26)
        self.assertAlmostEqual(ledger[2]["realized_pnl"], 446.0)
        self.assertAlmostEqual(ledger[2]["position_quantity_after"], 100.0)
        self.assertAlmostEqual(ledger[2]["avg_cost_after"], 12.26)
        self.assertEqual(daily[0]["trade_date"], "2026-05-21")
        self.assertAlmostEqual(daily[0]["realized_pnl"], 446.0)

    def test_paper_db_health_reports_missing_db_url(self) -> None:
        settings = SimpleNamespace(
            paper_db_url=None,
            futu_gateway_agent_id="aistockcn-paper-cn",
            futu_gateway_market="CN",
        )

        with mock.patch.object(paper_db_service, "get_settings", return_value=settings):
            result = paper_db_service.get_paper_db_health()

        self.assertFalse(result["healthy"])
        self.assertIn("PAPER_DB_URL", result["error"])

    def test_full_pipeline_rewrites_docker_db_host_for_host_network(self) -> None:
        self.assertEqual(
            pipeline_control_service._host_network_database_url(
                "postgresql://aistock_app:secret@postgres16:5432/aistock"
            ),
            "postgresql://aistock_app:secret@127.0.0.1:5432/aistock",
        )
        self.assertEqual(
            pipeline_control_service._host_network_database_url("postgresql://user:secret@db.example:5432/aistock"),
            "postgresql://user:secret@db.example:5432/aistock",
        )

    def test_paper_db_daily_history_contains_actual_trades_and_positions(self) -> None:
        fills = [
            {
                "created_at": datetime(2026, 5, 20, 1, 0),
                "broker_order_id": "buy-1",
                "symbol": "600703",
                "side": "BUY",
                "quantity": 200,
                "price": 10.0,
                "notional": 2000.0,
            },
            {
                "created_at": datetime(2026, 5, 21, 1, 0),
                "broker_order_id": "sell-1",
                "symbol": "600703",
                "side": "SELL",
                "quantity": 100,
                "price": 15.0,
                "notional": 1500.0,
            },
            {
                "created_at": datetime(2026, 5, 21, 1, 1),
                "broker_order_id": "buy-2",
                "symbol": "605288",
                "side": "BUY",
                "quantity": 100,
                "price": 50.0,
                "notional": 5000.0,
            },
        ]

        daily = paper_db_service.build_daily_position_history(fills, limit=10)

        self.assertEqual(daily[0]["trade_date"], "2026-05-21")
        self.assertEqual(daily[0]["fills_rows"], 2)
        self.assertEqual(daily[0]["positions_rows"], 2)
        latest_positions = {row["symbol"]: row for row in daily[0]["positions"]}
        self.assertAlmostEqual(latest_positions["600703"]["quantity"], 100.0)
        self.assertAlmostEqual(latest_positions["600703"]["avg_cost"], 10.0)
        self.assertAlmostEqual(latest_positions["600703"]["realized_pnl"], 500.0)
        self.assertAlmostEqual(latest_positions["605288"]["quantity"], 100.0)
        self.assertEqual(daily[1]["trade_date"], "2026-05-20")
        self.assertEqual(daily[1]["positions_rows"], 1)

    def test_valuation_uses_latest_prior_share_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            reference_dir = data_dir / "reference" / "valuation_reference"
            reference_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-25",
                        "code": "000001",
                        "total_market_cap": 200.0,
                        "float_market_cap": 150.0,
                        "total_shares": 10.0,
                        "float_shares": 8.0,
                    }
                ]
            ).to_parquet(reference_dir / "000001.parquet", index=False)

            bundle_df = pd.DataFrame(
                [
                    {"date": "2026-03-26", "code": "sz.000001", "close": 21.0, "pctChg": 1.0, "peTTM": 5, "pbMRQ": 1, "psTTM": 2, "pcfNcfTTM": 3},
                    {"date": "2026-03-27", "code": "sz.000001", "close": 22.0, "pctChg": 1.0, "peTTM": 5, "pbMRQ": 1, "psTTM": 2, "pcfNcfTTM": 3},
                ]
            )

            valuation_df, warning = build_valuation_df(
                bundle_df,
                "000001",
                start_date="20260326",
                end_date="20260327",
                data_dir=data_dir,
            )

            self.assertIn("reference_cache_stale_until:2026-03-25", warning or "")
            self.assertEqual(valuation_df["total_shares"].tolist(), [10.0, 10.0])
            self.assertEqual(valuation_df["float_shares"].tolist(), [8.0, 8.0])
            self.assertEqual(valuation_df["total_market_cap"].tolist(), [210.0, 220.0])
            self.assertEqual(valuation_df["float_market_cap"].tolist(), [168.0, 176.0])

    def test_reference_repair_overwrites_stale_non_null_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            reference_dir = data_dir / "reference" / "valuation_reference"
            valuation_dir = data_dir / "daily_valuation"
            reference_dir.mkdir(parents=True)
            valuation_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-21",
                        "code": "000001",
                        "total_market_cap": 200.0,
                        "float_market_cap": 100.0,
                        "total_shares": 20.0,
                        "float_shares": 10.0,
                    }
                ]
            ).to_parquet(reference_dir / "000001.parquet", index=False)
            valuation_path = valuation_dir / "000001.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-22",
                        "code": "000001",
                        "close": 3.0,
                        "total_market_cap": 54.0,
                        "float_market_cap": 21.0,
                        "total_shares": 18.0,
                        "float_shares": 7.0,
                    }
                ]
            ).to_parquet(valuation_path, index=False)

            changed, rows = repair_one(data_dir, valuation_path, overwrite_existing=True)
            repaired = pd.read_parquet(valuation_path)

        self.assertTrue(changed)
        self.assertEqual(rows, 1)
        self.assertEqual(repaired["total_shares"].tolist(), [20.0])
        self.assertEqual(repaired["float_shares"].tolist(), [10.0])
        self.assertEqual(repaired["total_market_cap"].tolist(), [60.0])
        self.assertEqual(repaired["float_market_cap"].tolist(), [30.0])

    def test_reference_status_allows_recent_slow_reference_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            reference_dir = data_dir / "reference" / "valuation_reference"
            reference_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-13",
                        "code": "000001",
                        "total_market_cap": 200.0,
                        "float_market_cap": 150.0,
                        "total_shares": 10.0,
                        "float_shares": 8.0,
                    }
                ]
            ).to_parquet(reference_dir / "000001.parquet", index=False)
            stock_df = pd.DataFrame([{"code": "000001", "exchange": "sz", "industry": "Bank"}])

            write_reference_status(data_dir, stock_df=stock_df, target_trade_date="2026-05-20")
            payload = json.loads(reference_status_path(data_dir).read_text(encoding="utf-8"))

            self.assertEqual(payload["industry_missing_count"], 0)
            self.assertEqual(payload["valuation_reference_ready_count"], 1)
            self.assertEqual(payload["valuation_reference_stale_count"], 0)
            self.assertEqual(payload["valuation_reference_stale_after_days"], 45)

    def test_reference_status_marks_old_slow_reference_cache_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            reference_dir = data_dir / "reference" / "valuation_reference"
            reference_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-01",
                        "code": "000001",
                        "total_market_cap": 200.0,
                        "float_market_cap": 150.0,
                        "total_shares": 10.0,
                        "float_shares": 8.0,
                    }
                ]
            ).to_parquet(reference_dir / "000001.parquet", index=False)
            stock_df = pd.DataFrame([{"code": "000001", "exchange": "sz", "industry": "Bank"}])

            write_reference_status(data_dir, stock_df=stock_df, target_trade_date="2026-05-20")
            payload = json.loads(reference_status_path(data_dir).read_text(encoding="utf-8"))

            self.assertEqual(payload["valuation_reference_ready_count"], 0)
            self.assertEqual(payload["valuation_reference_stale_count"], 1)

    def test_incremental_merge_preserves_existing_non_null_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valuation.parquet"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-25",
                        "code": "000001",
                        "close": 10.0,
                        "total_market_cap": 100.0,
                        "total_shares": 10.0,
                    }
                ]
            ).to_parquet(path, index=False)
            fresh_df = pd.DataFrame(
                [
                    {
                        "date": "2026-03-25",
                        "code": "000001",
                        "close": 11.0,
                        "total_market_cap": pd.NA,
                        "total_shares": pd.NA,
                    }
                ]
            )

            merged = merge_existing_output(path, fresh_df)

            self.assertEqual(float(merged.loc[0, "close"]), 11.0)
            self.assertEqual(float(merged.loc[0, "total_market_cap"]), 100.0)
            self.assertEqual(float(merged.loc[0, "total_shares"]), 10.0)

    def test_batch_does_not_mark_short_history_current_by_max_date_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kline_dir = root / "daily_kline"
            valuation_dir = root / "daily_valuation"
            kline_dir.mkdir()
            valuation_dir.mkdir()
            short_history = pd.DataFrame(
                [
                    {"date": "2026-01-05", "code": "000001", "close": 10.0},
                    {"date": "2026-05-22", "code": "000001", "close": 11.0},
                ]
            )
            short_history.to_parquet(kline_dir / "000001.parquet", index=False)
            short_history.to_parquet(valuation_dir / "000001.parquet", index=False)
            state = {"done_codes": []}

            mark_existing_outputs(
                ["000001"],
                state=state,
                kline_dir=kline_dir,
                valuation_dir=valuation_dir,
                overwrite=False,
                start_date="20230322",
                target_trade_date="2026-05-22",
            )

        self.assertEqual(state["done_codes"], [])

    def test_batch_marks_full_history_current_when_min_and_max_cover_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kline_dir = root / "daily_kline"
            valuation_dir = root / "daily_valuation"
            kline_dir.mkdir()
            valuation_dir.mkdir()
            full_history = pd.DataFrame(
                [
                    {"date": "2023-03-22", "code": "000001", "close": 10.0},
                    {"date": "2026-05-22", "code": "000001", "close": 11.0},
                ]
            )
            full_history.to_parquet(kline_dir / "000001.parquet", index=False)
            full_history.to_parquet(valuation_dir / "000001.parquet", index=False)
            state = {"done_codes": []}

            mark_existing_outputs(
                ["000001"],
                state=state,
                kline_dir=kline_dir,
                valuation_dir=valuation_dir,
                overwrite=False,
                start_date="20230322",
                target_trade_date="2026-05-22",
            )

        self.assertEqual(state["done_codes"], ["000001"])

    def test_process_code_refetches_from_start_for_truncated_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kline_dir = root / "daily_kline"
            valuation_dir = root / "daily_valuation"
            kline_dir.mkdir()
            valuation_dir.mkdir()
            existing = pd.DataFrame(
                [
                    {"date": "2026-01-05", "code": "000001", "close": 10.0},
                    {"date": "2026-05-22", "code": "000001", "close": 11.0},
                ]
            )
            existing.to_parquet(kline_dir / "000001.parquet", index=False)
            existing.to_parquet(valuation_dir / "000001.parquet", index=False)
            fresh = pd.DataFrame(
                [
                    {"date": "2023-03-22", "code": "000001", "close": 8.0},
                    {"date": "2026-05-22", "code": "000001", "close": 11.0},
                ]
            )
            args = SimpleNamespace(start_date="20230322", end_date="20260522", overwrite=False)

            with (
                mock.patch.object(batch_download, "download_baostock_daily_bundle", return_value=(fresh, None)) as download_mock,
                mock.patch.object(batch_download, "build_kline_df", return_value=fresh),
                mock.patch.object(batch_download, "build_valuation_df", return_value=(fresh, None)),
            ):
                ok, reason = process_code(
                    "000001",
                    exchange="sz",
                    args=args,
                    data_dir=root,
                    kline_dir=kline_dir,
                    valuation_dir=valuation_dir,
                    target_trade_date="2026-05-22",
                )

        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(download_mock.call_args.kwargs["start_date"], "20230322")

    def test_inference_keeps_rows_with_only_recoverable_reference_nulls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "daily_kline").mkdir(parents=True)
            (data_dir / "daily_valuation").mkdir(parents=True)
            dates = pd.date_range("2026-03-18", periods=25, freq="B")
            pd.DataFrame([{"code": "000001", "exchange": "sz", "name": "Ping An", "industry": "Bank"}]).to_parquet(
                data_dir / "stock_list.parquet",
                index=False,
            )
            pd.DataFrame(
                {
                    "date": dates,
                    "code": "000001",
                    "exchange": "sz",
                    "open": range(10, 35),
                    "high": range(11, 36),
                    "low": range(9, 34),
                    "close": range(10, 35),
                    "volume": [1000] * len(dates),
                    "amount": [10000] * len(dates),
                    "turnover": [1.0] * len(dates),
                    "amplitude": [1.0] * len(dates),
                    "pct_chg": [0.1] * len(dates),
                    "change": [0.1] * len(dates),
                }
            ).to_parquet(data_dir / "daily_kline" / "000001.parquet", index=False)
            pd.DataFrame(
                {
                    "date": dates,
                    "code": "000001",
                    "exchange": "sz",
                    "close": range(10, 35),
                    "pct_chg": [0.1] * len(dates),
                    "total_market_cap": [pd.NA] * len(dates),
                    "float_market_cap": [pd.NA] * len(dates),
                    "total_shares": [pd.NA] * len(dates),
                    "float_shares": [pd.NA] * len(dates),
                    "pe_ttm": [5.0] * len(dates),
                    "pb": [1.0] * len(dates),
                    "ps": [2.0] * len(dates),
                    "pcf": [3.0] * len(dates),
                }
            ).to_parquet(data_dir / "daily_valuation" / "000001.parquet", index=False)

            inference_df = build_inference_frame(data_dir, limit=0, as_of_date=None)

            self.assertEqual(len(inference_df), 1)
            self.assertEqual(inference_df.loc[0, "code"], "000001")

    def test_st_name_filter_and_control_settings_default_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp)
            settings = read_control_settings(quant_dir)

        self.assertTrue(settings["exclude_st_from_model_candidates"])
        self.assertTrue(is_st_stock_name("*ST元道"))
        self.assertTrue(is_st_stock_name("ST金顶"))
        self.assertTrue(is_st_stock_name("S*ST测试"))
        self.assertFalse(is_st_stock_name("平安银行"))
        self.assertFalse(is_investable_stock_name("退市观典", exclude_st=False))
        self.assertFalse(is_investable_stock_name("退市股退", exclude_st=False))
        self.assertFalse(is_investable_stock_name("泽达退", exclude_st=False))
        self.assertFalse(is_investable_stock_name("ST金顶", exclude_st=True))
        self.assertTrue(is_investable_stock_name("ST金顶", exclude_st=False))
        self.assertFalse(is_universe_investable_stock_name("退市观典"))
        self.assertFalse(is_universe_investable_stock_name("泽达退"))
        self.assertTrue(is_universe_investable_stock_name("ST金顶"))

    def test_inference_candidate_loading_excludes_st_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "daily_kline").mkdir(parents=True)
            (data_dir / "daily_valuation").mkdir(parents=True)
            dates = pd.date_range("2026-03-18", periods=25, freq="B")
            pd.DataFrame(
                [
                    {"code": "000001", "exchange": "sz", "name": "平安银行", "industry": "Bank"},
                    {"code": "688496", "exchange": "sh", "name": "*ST清越", "industry": "Display"},
                ]
            ).to_parquet(data_dir / "stock_list.parquet", index=False)
            for code, exchange in [("000001", "sz"), ("688496", "sh")]:
                pd.DataFrame(
                    {
                        "date": dates,
                        "code": code,
                        "exchange": exchange,
                        "open": range(10, 35),
                        "high": range(11, 36),
                        "low": range(9, 34),
                        "close": range(10, 35),
                        "volume": [1000] * len(dates),
                        "amount": [10000] * len(dates),
                        "turnover": [1.0] * len(dates),
                        "amplitude": [1.0] * len(dates),
                        "pct_chg": [0.1] * len(dates),
                        "change": [0.1] * len(dates),
                    }
                ).to_parquet(data_dir / "daily_kline" / f"{code}.parquet", index=False)
                pd.DataFrame(
                    {
                        "date": dates,
                        "code": code,
                        "exchange": exchange,
                        "close": range(10, 35),
                        "pct_chg": [0.1] * len(dates),
                        "total_market_cap": [100.0] * len(dates),
                        "float_market_cap": [90.0] * len(dates),
                        "total_shares": [10.0] * len(dates),
                        "float_shares": [9.0] * len(dates),
                        "pe_ttm": [5.0] * len(dates),
                        "pb": [1.0] * len(dates),
                        "ps": [2.0] * len(dates),
                        "pcf": [3.0] * len(dates),
                    }
                ).to_parquet(data_dir / "daily_valuation" / f"{code}.parquet", index=False)

            enabled_df = build_inference_frame(data_dir, limit=0, as_of_date=None)
            write_control_settings(data_dir, {"exclude_st_from_model_candidates": False})
            disabled_df = build_inference_frame(data_dir, limit=0, as_of_date=None)

        self.assertEqual(enabled_df["code"].tolist(), ["000001"])
        self.assertEqual(disabled_df["code"].tolist(), ["000001", "688496"])

    def test_model_picks_filters_st_from_old_score_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            models_dir = quant_dir / "models"
            models_dir.mkdir(parents=True)
            scores_path = models_dir / "inference_scores_latest.parquet"
            pd.DataFrame(
                [
                    {"date": "2026-05-20", "code": "688287", "name": "退市观典", "industry": "Defense", "score": 1.0, "close": 0.45},
                    {"date": "2026-05-20", "code": "688496", "name": "*ST清越", "industry": "Display", "score": 0.99, "close": 1.5},
                    {"date": "2026-05-20", "code": "000001", "name": "平安银行", "industry": "Bank", "score": 0.8, "close": 10.0},
                ]
            ).to_parquet(scores_path, index=False)
            pd.DataFrame([{"date": "2026-05-20"}]).to_parquet(quant_dir / "inference_features_latest.parquet", index=False)
            settings = SimpleNamespace(
                quant_dir=quant_dir,
                models_dir=models_dir,
                stock_list_path=quant_dir / "missing_stock_list.parquet",
                stock_registry_path=quant_dir / "missing_stock_registry.parquet",
                control_settings_path=quant_dir / "control_settings.json",
            )

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(admin_settings_service, "get_settings", return_value=settings):
                    with mock.patch.object(model_service, "_scores_path_for_profile", return_value=(None, scores_path)):
                        result = model_service.get_latest_picks(limit=5)

        self.assertEqual(result["rows"], 1)
        self.assertEqual([row["code"] for row in result["picks"]], ["000001"])

    def test_provider_probe_timeout_is_reported(self) -> None:
        with mock.patch.object(source_readiness, "bs", object()), mock.patch.object(source_readiness, "ak", object()):
            with mock.patch.object(
                source_readiness,
                "_run_provider_probe",
                side_effect=source_readiness.ProviderProbeTimeoutError("provider probe timed out after 30s"),
            ):
                result = source_readiness.get_china_market_data_readiness(local_date="2026-05-12")

        self.assertEqual(result["reason"], "baostock_probe_timeout")
        self.assertIn("timed out", result["baostock"]["error"])

    def test_benchmark_history_normalizes_akshare_index_schema(self) -> None:
        raw = pd.DataFrame(
            [
                {"日期": "2026-05-18", "开盘": "3900.1", "收盘": "3910.2", "成交量": "100"},
                {"日期": "2026-05-19", "开盘": "3910.2", "收盘": "3920.3", "成交量": "110"},
            ]
        )

        normalized = benchmark_service.normalize_akshare_index_history(raw)

        self.assertEqual(normalized["code"].tolist(), ["000300.SH", "000300.SH"])
        self.assertEqual(normalized["source"].tolist(), ["akshare.index_zh_a_hist", "akshare.index_zh_a_hist"])
        self.assertEqual(normalized["close"].tolist(), [3910.2, 3920.3])

    def test_benchmark_refresh_merges_into_canonical_index_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            index_dir = quant_dir / "index"
            index_dir.mkdir(parents=True)
            path = index_dir / "000300.SH.parquet"
            pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2026-05-18"),
                        "code": "000300.SH",
                        "name": "沪深300",
                        "close": 3910.2,
                        "source": "akshare.index_zh_a_hist",
                        "updated_at": "2026-05-18T00:00:00+00:00",
                    }
                ]
            ).to_parquet(path, index=False)

            fake_ak = SimpleNamespace(
                index_zh_a_hist=mock.Mock(
                    return_value=pd.DataFrame(
                        [
                            {"日期": "2026-05-18", "收盘": 3911.0},
                            {"日期": "2026-05-19", "收盘": 3920.3},
                        ]
                    )
                )
            )
            settings = SimpleNamespace(quant_dir=quant_dir)

            with mock.patch.object(benchmark_service, "get_settings", return_value=settings):
                with mock.patch.object(benchmark_service, "_load_akshare", return_value=fake_ak):
                    result = benchmark_service.refresh_benchmark_history(end_date="20260519")

            stored = pd.read_parquet(path).sort_values("date").reset_index(drop=True)
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["latest_date"], "2026-05-19")
            self.assertEqual(stored["close"].tolist(), [3911.0, 3920.3])

    def test_recent_state_file_mtime_prevents_false_stall_on_json_race(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            logs_dir = root / "logs"
            quant_dir = root / "quant_data"
            run_dir.mkdir()
            logs_dir.mkdir()
            (quant_dir / "batch_state").mkdir(parents=True)

            state_file = quant_dir / "batch_state" / "all_a_3y_state.json"
            state_file.write_text("{", encoding="utf-8")
            (run_dir / "full_market_3y.pid").write_text("container-1\n", encoding="utf-8")
            log_file = logs_dir / "full_market_3y_20260513T100045Z.log"
            log_file.write_text("started\n", encoding="utf-8")
            old_ts = (datetime.now(timezone.utc) - timedelta(minutes=60)).timestamp()
            log_file.touch()
            import os

            os.utime(log_file, (old_ts, old_ts))

            settings = SimpleNamespace(
                state_file=state_file,
                run_dir=run_dir,
                logs_dir=logs_dir,
                stock_list_path=quant_dir / "stock_list.parquet",
            )
            container = {
                "container_id": "container-1",
                "container_name": "aistockcn-full-market-3y-test",
                "status": "running",
                "running_for": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "oom_killed": False,
                "is_running": True,
            }

            with mock.patch.object(batch_service, "get_settings", return_value=settings):
                with mock.patch.object(batch_service, "_get_container_info", return_value=container):
                    status = batch_service.get_batch_status()

        self.assertFalse(status["is_stalled"])
        self.assertEqual(status["state_file_updated_at"], status["last_activity_at"])

    def test_daily_batch_defaults_refresh_only_current_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_file = root / "quant_data" / "batch_state" / "all_a_3y_state.json"
            state_file.parent.mkdir(parents=True)
            state_file.write_text(json.dumps({"start_date": "20230322"}), encoding="utf-8")
            settings = SimpleNamespace(state_file=state_file)

            with mock.patch.object(batch_service, "get_settings", return_value=settings):
                with mock.patch.object(batch_service, "_china_today", return_value=date(2026, 6, 16)):
                    defaults = batch_service._default_batch_args()

        self.assertEqual(defaults["start_date"], "20260616")
        self.assertEqual(defaults["end_date"], "20260616")

    def test_paper_targets_hide_noop_zero_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_path = Path(tmp) / "targets_latest.parquet"
            pd.DataFrame(
                [
                    {
                        "signal_date": "2026-05-19",
                        "rank": None,
                        "code": "000010",
                        "name": None,
                        "score": None,
                        "close": 2.73,
                        "target_qty": 0,
                        "current_qty": 0,
                        "delta_qty": None,
                        "buy_order_qty": 0,
                        "sell_order_qty": 0,
                        "action": None,
                        "current_market_value": 0,
                    },
                    {
                        "signal_date": "2026-05-19",
                        "rank": 1,
                        "code": "688496",
                        "name": "*ST清越",
                        "score": 0.89,
                        "close": 1.51,
                        "target_qty": 100,
                        "current_qty": 0,
                        "delta_qty": 100,
                        "buy_order_qty": 100,
                        "sell_order_qty": 0,
                        "action": "BUY",
                        "current_market_value": 0,
                    },
                    {
                        "signal_date": "2026-05-19",
                        "rank": None,
                        "code": "002294",
                        "name": "002294",
                        "score": None,
                        "close": 38.96,
                        "target_qty": 0,
                        "current_qty": 100,
                        "delta_qty": -100,
                        "buy_order_qty": 0,
                        "sell_order_qty": 100,
                        "action": "SELL",
                        "current_market_value": 3896,
                    },
                ]
            ).to_parquet(targets_path, index=False)
            settings = SimpleNamespace(paper_trading_targets_path=targets_path)

            with mock.patch.object(paper_service, "get_settings", return_value=settings):
                result = paper_service.get_paper_trading_targets(limit=10)

        self.assertEqual(result["rows"], 2)
        self.assertEqual([row["code"] for row in result["targets"]], ["688496", "002294"])

    def test_paper_target_persistence_drops_noop_zero_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            targets_path = Path(tmp) / "targets_latest.parquet"
            plan = pd.DataFrame(
                [
                    {
                        "signal_date": "2026-05-19",
                        "rank": None,
                        "code": "000010",
                        "score": None,
                        "close": 2.73,
                        "target_qty": 0,
                        "current_qty": 0,
                        "delta_qty": None,
                        "sell_order_qty": 0,
                        "buy_order_qty": 0,
                        "action": "HOLD",
                        "current_market_value": 0,
                    },
                    {
                        "signal_date": "2026-05-19",
                        "rank": 1,
                        "code": "688496",
                        "score": 0.89,
                        "close": 1.51,
                        "target_qty": 100,
                        "current_qty": 0,
                        "delta_qty": 100,
                        "sell_order_qty": 0,
                        "buy_order_qty": 100,
                        "action": "BUY",
                        "current_market_value": 0,
                    },
                ]
            )

            persist_targets({"targets": targets_path}, plan)
            stored = pd.read_parquet(targets_path)

        self.assertEqual(stored["code"].tolist(), ["688496"])

    def test_paper_history_filters_noop_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "sync_history.jsonl"
            rows = [
                {"status": "noop", "recorded_at": "2026-05-20T01:00:00+00:00"},
                {"status": "success", "recorded_at": "2026-05-20T01:01:00+00:00", "message": "placed"},
                {"status": "dry_run", "recorded_at": "2026-05-20T01:02:00+00:00"},
                {"status": "error", "recorded_at": "2026-05-20T01:03:00+00:00", "message": "failed"},
            ]
            history_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            settings = SimpleNamespace(paper_trading_history_path=history_path)

            with mock.patch.object(paper_service, "get_settings", return_value=settings):
                history = paper_service.get_paper_trading_history(limit=10)
                performance = paper_service.get_paper_trading_performance(limit=10)

        self.assertEqual(history["rows"], 2)
        self.assertEqual([row["status"] for row in history["history"]], ["error", "success"])
        self.assertEqual(performance["rows"], 2)
        self.assertEqual([row["status"] for row in performance["snapshots"]], ["success", "error"])

    def test_paper_positions_filter_closed_zero_quantity_rows(self) -> None:
        class PositionClient:
            def __init__(self, settings: object) -> None:
                self.settings = settings

            def get_positions(self) -> list[dict[str, object]]:
                return [
                    {"symbol": "000010", "quantity": 0, "last_price": 2.73, "market_value": 0},
                    {"symbol": "000001", "quantity": 200, "last_price": 10.0, "market_value": 2000.0},
                    {"symbol": "000002", "qty": "0", "last_price": 9.0, "market_val": 0},
                ]

        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(
                stock_list_path=Path(tmp) / "missing_stock_list.parquet",
                stock_registry_path=Path(tmp) / "missing_stock_registry.parquet",
            )
            with mock.patch.object(paper_service, "get_settings", return_value=settings):
                with mock.patch.object(paper_service, "PaperGatewayClient", PositionClient):
                    result = paper_service.get_paper_trading_positions(limit=20)

        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["positions"][0]["symbol"], "000001")
        self.assertEqual(result["positions"][0]["quantity"], 200.0)

    def test_daily_snapshot_replaces_same_trade_date_and_filters_closed_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "daily_snapshots.json"
            write_daily_snapshot(
                path,
                trade_date="2026-05-21",
                balance_metrics={"cash": 1000.0, "total_assets": 1500.0, "currency": "CN"},
                live_summary={"market_value": 500.0, "total_pnl": 10.0},
                positions=[
                    {"symbol": "000010", "quantity": 0, "market_value": 0, "last_price": 2.73},
                    {"symbol": "000001", "quantity": 100, "market_value": 1000.0, "last_price": 10.0},
                ],
                orders=[{"broker_order_id": "old", "symbol": "000001", "created_at": "2026-05-21 09:40:00"}],
            )
            write_daily_snapshot(
                path,
                trade_date="2026-05-21",
                balance_metrics={"cash": 900.0, "total_assets": 1600.0, "currency": "CN"},
                live_summary={"market_value": 700.0, "total_pnl": 12.0},
                positions=[
                    {"symbol": "000010", "quantity": 0, "market_value": 0, "last_price": 2.73},
                    {"symbol": "000002", "quantity": 200, "market_value": 1200.0, "last_price": 6.0},
                ],
                orders=[
                    {"broker_order_id": "new", "symbol": "000002", "created_at": "2026-05-21T02:07:31"},
                    {"broker_order_id": "next-day", "symbol": "000003", "created_at": "2026-05-22 09:40:00"},
                ],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(list(payload.keys()), ["2026-05-21"])
        snapshot = payload["2026-05-21"]
        self.assertEqual(snapshot["summary"]["cash"], 900.0)
        self.assertEqual(snapshot["positions_rows"], 1)
        self.assertEqual(snapshot["positions"][0]["symbol"], "000002")
        self.assertEqual(snapshot["orders_rows"], 1)
        self.assertEqual(snapshot["orders"][0]["broker_order_id"], "new")

    def test_daily_history_api_merges_snapshots_and_history_by_trade_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshots_path = root / "daily_snapshots.json"
            history_path = root / "sync_history.jsonl"
            snapshots_path.write_text(
                json.dumps(
                    {
                        "2026-05-21": {
                            "trade_date": "2026-05-21",
                            "generated_at": "2026-05-21T03:00:00+00:00",
                            "summary": {"total_assets": 1000.0},
                            "positions": [{"symbol": "000001", "quantity": 100, "market_value": 1000.0}],
                            "positions_rows": 1,
                            "orders": [],
                            "orders_rows": 0,
                            "positions_snapshot_available": True,
                        },
                        "2026-05-19": {
                            "trade_date": "2026-05-19",
                            "generated_at": "2026-05-19T03:00:00+00:00",
                            "summary": {"total_assets": 700.0},
                            "positions": [],
                            "positions_rows": 0,
                            "orders": [],
                            "orders_rows": 0,
                            "positions_snapshot_available": False,
                        }
                    }
                ),
                encoding="utf-8",
            )
            history_path.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "recorded_at": "2026-05-20T18:40:00+00:00",
                        "balance_metrics": {"cash": 800.0, "total_assets": 900.0},
                        "live_summary": {"market_value": 100.0, "total_pnl": 5.0},
                        "placed_orders": [
                            {
                                "broker_order_id": "order-1",
                                "symbol": "600000",
                                "side": "BUY",
                                "created_at": "2026-05-20T18:30:00+00:00",
                            }
                        ],
                        "skipped_orders": [
                            {
                                "symbol": "002294",
                                "side": "SELL",
                                "quantity": 100,
                                "error": "quote unavailable",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            settings = SimpleNamespace(
                paper_trading_daily_snapshots_path=snapshots_path,
                paper_trading_history_path=history_path,
            )
            with mock.patch.object(paper_service, "get_settings", return_value=settings):
                result = paper_service.get_paper_trading_daily_history(limit=20)

        self.assertEqual([row["trade_date"] for row in result["daily"]], ["2026-05-21", "2026-05-19"])
        row = result["daily"][0]
        self.assertTrue(row["positions_snapshot_available"])
        self.assertEqual(row["positions_rows"], 1)
        self.assertEqual(row["orders_rows"], 1)
        self.assertEqual(row["orders"][0]["broker_order_id"], "order-1")
        self.assertNotEqual(row["orders"][0]["symbol"], "002294")

    def test_paper_sync_noop_updates_state_without_ledger_entry(self) -> None:
        class NoopGateway:
            def __init__(self, config: SyncConfig) -> None:
                self.config = config

            def health(self) -> dict[str, object]:
                return {"status": "ok"}

            def sync_agent(self) -> None:
                return None

            def get_agent_positions(self) -> list[dict[str, object]]:
                return []

            def get_agent_orders(self) -> list[dict[str, object]]:
                return []

            def get_balance(self) -> list[dict[str, object]]:
                return [{"cash": 1000.0, "power": 1000.0, "total_assets": 1000.0}]

            def get_agent_summary(self) -> dict[str, object]:
                return {"total_assets": 1000.0, "total_pnl": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-20",
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "Ping An Bank",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                        "amount": 100_000_000.0,
                        "pct_chg": 0.0,
                        "change": 0.0,
                    }
                ]
            ).to_parquet(scores_path, index=False)
            signature = score_file_signature(scores_path)
            (state_dir / "state.json").write_text(
                json.dumps({"last_score_signature": signature}),
                encoding="utf-8",
            )
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=state_dir,
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=None,
                max_buy_order_qty=0,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=False,
            )

            with mock.patch("paper_trade_futu.GatewayClient", NoopGateway):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(pd.DataFrame(), {"buy_order_count": 0, "sell_order_count": 0}),
                ):
                    code, state = sync_once(config)

            history_path = state_dir / "sync_history.jsonl"

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "noop")
        self.assertFalse(history_path.exists())

    def test_paper_sync_deduplicates_repeated_identical_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_dir = root / "state"
            state_dir.mkdir()
            config = self._paper_sync_config(root / "missing.parquet", state_dir)

            with mock.patch("paper_trade_futu.resolve_paper_model", side_effect=RuntimeError("artifact mismatch")):
                first_code, first_state = sync_once(config)
                second_code, second_state = sync_once(config)

            history = (state_dir / "sync_history.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(first_code, 1)
        self.assertEqual(second_code, 1)
        self.assertEqual(first_state["last_error_repeat_count"], 1)
        self.assertEqual(second_state["last_error_repeat_count"], 2)
        self.assertEqual(len(history), 1)

    def test_paper_model_path_must_be_immutable(self) -> None:
        from paper_trade_futu import verify_immutable_model_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = {
                "market": "CN",
                "model_version": "cn-medium-v1",
                "artifact_path": "quant_data/model_profiles/medium/models",
            }
            with self.assertRaisesRegex(RuntimeError, "uses a mutable artifact path"):
                verify_immutable_model_path(root, model)

            expected = root / "quant_data" / "model_registry" / "CN" / "cn-medium-v1"
            model["artifact_path"] = str(expected.relative_to(root))
            self.assertEqual(verify_immutable_model_path(root, model), expected)

    def test_paper_sync_does_not_reorder_same_score_snapshot(self) -> None:
        class NoopGateway:
            def __init__(self, config: SyncConfig) -> None:
                self.config = config

            def health(self) -> dict[str, object]:
                return {"status": "ok"}

            def sync_agent(self) -> None:
                return None

            def get_agent_positions(self) -> list[dict[str, object]]:
                return []

            def get_agent_orders(self) -> list[dict[str, object]]:
                return []

            def get_balance(self) -> list[dict[str, object]]:
                return [{"cash": 1000.0, "power": 1000.0, "total_assets": 1000.0}]

            def get_agent_summary(self) -> dict[str, object]:
                return {"total_assets": 1000.0, "total_pnl": 0.0}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-20",
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "Ping An Bank",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                        "amount": 100_000_000.0,
                        "pct_chg": 0.0,
                        "change": 0.0,
                    }
                ]
            ).to_parquet(scores_path, index=False)
            signature = score_file_signature(scores_path)
            (state_dir / "state.json").write_text(
                json.dumps({"last_score_signature": signature}),
                encoding="utf-8",
            )
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=state_dir,
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=None,
                max_buy_order_qty=0,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=False,
            )
            pending_plan = pd.DataFrame(
                [
                    {
                        "code": "000001",
                        "rank": 1,
                        "score": 0.9,
                        "buy_order_qty": 100,
                        "sell_order_qty": 0,
                    }
                ]
            )

            with mock.patch("paper_trade_futu.GatewayClient", NoopGateway):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(pending_plan, {"buy_order_count": 1, "sell_order_count": 0}),
                ):
                    with mock.patch("paper_trade_futu.execute_plan") as execute:
                        code, state = sync_once(config)

            history_path = state_dir / "sync_history.jsonl"

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "noop")
        self.assertIn("already been attempted", state["last_message"])
        self.assertFalse(history_path.exists())
        execute.assert_not_called()

    def test_paper_sync_waits_for_5d_rebalance_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            self._write_scores(scores_path, ["2026-05-26"])
            signature = score_file_signature(scores_path)
            (state_dir / "state.json").write_text(
                json.dumps(
                    {
                        "last_applied_signal_date": "2026-05-20",
                        "rebalance_observed_signal_dates": ["2026-05-21", "2026-05-22", "2026-05-25"],
                    }
                ),
                encoding="utf-8",
            )
            config = self._paper_sync_config(scores_path, state_dir)

            with mock.patch("paper_trade_futu.GatewayClient", self._noop_gateway_class()):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(self._pending_plan(), {"buy_order_count": 1, "sell_order_count": 0}),
                ):
                    with mock.patch("paper_trade_futu.execute_plan") as execute:
                        code, state = sync_once(config)

            history_path = state_dir / "sync_history.jsonl"

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "noop")
        self.assertFalse(state["rebalance_due"])
        self.assertEqual(state["rebalance_every"], 5)
        self.assertEqual(state["rebalance_wait_count"], 4)
        self.assertEqual(
            state["rebalance_observed_signal_dates"],
            ["2026-05-21", "2026-05-22", "2026-05-25", "2026-05-26"],
        )
        self.assertEqual(state["last_applied_signal_date"], "2026-05-20")
        self.assertEqual(state["last_score_signature"], signature)
        self.assertIn("not due", state["last_message"])
        self.assertFalse(history_path.exists())
        execute.assert_not_called()

    def test_paper_sync_trades_on_5d_rebalance_window(self) -> None:
        execution_rows = self._pending_plan().assign(
            sent_order_id=None,
            sent_status=None,
            sent_price=None,
            sent_reference_price=None,
            sent_error=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            self._write_scores(scores_path, ["2026-05-27"])
            (state_dir / "state.json").write_text(
                json.dumps(
                    {
                        "last_applied_signal_date": "2026-05-20",
                        "rebalance_observed_signal_dates": ["2026-05-21", "2026-05-22", "2026-05-25", "2026-05-26"],
                    }
                ),
                encoding="utf-8",
            )
            config = self._paper_sync_config(scores_path, state_dir)

            with mock.patch("paper_trade_futu.GatewayClient", self._noop_gateway_class()):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(self._pending_plan(), {"buy_order_count": 1, "sell_order_count": 0}),
                ):
                    with mock.patch(
                        "paper_trade_futu.execute_plan",
                        return_value={
                            "execution_rows": execution_rows,
                            "cancelled_orders": [],
                            "placed_orders": [{"order_id": "order-1"}],
                            "skipped_orders": [],
                            "execution_skipped": False,
                        },
                    ) as execute:
                        code, state = sync_once(config)

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "success")
        self.assertTrue(state["rebalance_due"])
        self.assertEqual(state["rebalance_wait_count"], 5)
        self.assertEqual(state["rebalance_observed_signal_dates"], [])
        self.assertEqual(state["last_applied_signal_date"], "2026-05-27")
        execute.assert_called_once()

    def test_paper_sync_1d_profile_trades_each_new_score_date(self) -> None:
        execution_rows = self._pending_plan().assign(
            sent_order_id=None,
            sent_status=None,
            sent_price=None,
            sent_reference_price=None,
            sent_error=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            self._write_scores(
                scores_path,
                ["2026-05-20", "2026-05-21"],
                profile_name="short_1d",
                label_horizon=1,
            )
            (state_dir / "state.json").write_text(
                json.dumps({"last_applied_signal_date": "2026-05-20"}),
                encoding="utf-8",
            )
            config = self._paper_sync_config(scores_path, state_dir)

            with mock.patch("paper_trade_futu.GatewayClient", self._noop_gateway_class()):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(self._pending_plan(), {"buy_order_count": 1, "sell_order_count": 0}),
                ):
                    with mock.patch(
                        "paper_trade_futu.execute_plan",
                        return_value={
                            "execution_rows": execution_rows,
                            "cancelled_orders": [],
                            "placed_orders": [{"order_id": "order-1"}],
                            "skipped_orders": [],
                            "execution_skipped": False,
                        },
                    ) as execute:
                        code, state = sync_once(config)

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "success")
        self.assertEqual(state["rebalance_every"], 1)
        self.assertTrue(state["rebalance_due"])
        execute.assert_called_once()

    def test_paper_sync_force_overrides_rebalance_wait(self) -> None:
        execution_rows = self._pending_plan().assign(
            sent_order_id=None,
            sent_status=None,
            sent_price=None,
            sent_reference_price=None,
            sent_error=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores_path = root / "scores.parquet"
            state_dir = root / "state"
            state_dir.mkdir()
            self._write_scores(scores_path, ["2026-05-20", "2026-05-21"])
            (state_dir / "state.json").write_text(
                json.dumps({"last_applied_signal_date": "2026-05-20"}),
                encoding="utf-8",
            )
            config = self._paper_sync_config(scores_path, state_dir, force=True)

            with mock.patch("paper_trade_futu.GatewayClient", self._noop_gateway_class()):
                with mock.patch(
                    "paper_trade_futu.build_plan",
                    return_value=(self._pending_plan(), {"buy_order_count": 1, "sell_order_count": 0}),
                ):
                    with mock.patch(
                        "paper_trade_futu.execute_plan",
                        return_value={
                            "execution_rows": execution_rows,
                            "cancelled_orders": [],
                            "placed_orders": [{"order_id": "order-1"}],
                            "skipped_orders": [],
                            "execution_skipped": False,
                        },
                    ) as execute:
                        code, state = sync_once(config)

        self.assertEqual(code, 0)
        self.assertEqual(state["last_status"], "success")
        self.assertTrue(state["rebalance_due"])
        self.assertEqual(state["rebalance_wait_count"], 1)
        execute.assert_called_once()

    def test_transaction_fee_model_matches_a_share_rules(self) -> None:
        self.assertAlmostEqual(transaction_fee("BUY", 10_000.0), 20.4, places=6)
        self.assertAlmostEqual(transaction_fee("SELL", 10_000.0), 25.4, places=6)
        self.assertEqual(transaction_fee("BUY", 0.0), 0.0)

    def test_conservative_execution_model_price_limit_and_liquidity_rules(self) -> None:
        self.assertAlmostEqual(board_price_limit_rate("000001", "平安银行"), 0.10)
        self.assertAlmostEqual(board_price_limit_rate("300001", "创业板"), 0.20)
        self.assertAlmostEqual(board_price_limit_rate("688001", "科创板"), 0.20)
        self.assertAlmostEqual(board_price_limit_rate("688496", "*ST清越"), 0.05)
        self.assertAlmostEqual(liquidity_cap_notional(100_000_000.0), 500_000.0)
        self.assertEqual(
            buy_liquidity_skip_reason(amount=10_000_000.0, order_notional=10_000.0),
            "SKIP_LOW_LIQUIDITY",
        )
        self.assertEqual(
            buy_liquidity_skip_reason(amount=100_000_000.0, order_notional=600_000.0),
            "SKIP_LIQUIDITY_CAP",
        )
        self.assertIsNone(buy_liquidity_skip_reason(amount=100_000_000.0, order_notional=10_000.0))
        self.assertTrue(
            near_price_limit(
                side="BUY",
                price=10.96,
                previous_close=10.0,
                symbol="000001",
                name="平安银行",
            )
        )
        self.assertTrue(
            near_price_limit(
                side="SELL",
                price=9.04,
                previous_close=10.0,
                symbol="000001",
                name="平安银行",
            )
        )

    def test_paper_plan_reserves_buy_fees_before_sizing_order(self) -> None:
        affordable = compute_affordable_buy_quantity(cash_available=1020.0, price=10.0, lot_size=100)
        self.assertEqual(affordable, 0)
        affordable = compute_affordable_buy_quantity(cash_available=1021.0, price=10.0, lot_size=100)
        self.assertEqual(affordable, 100)

    def test_paper_plan_records_estimated_order_fees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = SyncConfig(
                scores_path=Path(tmp) / "scores.parquet",
                state_dir=Path(tmp),
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=None,
                max_buy_order_qty=1000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )
            latest_scores = pd.DataFrame(
                [
                    {
                        "date": "2026-05-19",
                        "rank": 1,
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "Ping An Bank",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                        "amount": 100_000_000.0,
                        "pct_chg": 0.0,
                        "change": 0.0,
                    }
                ]
            )

            plan, summary = build_plan(
                config,
                latest_scores=latest_scores,
                positions={},
                balance_metrics={"power": 1021.0, "cash": 1021.0, "total_assets": 1021.0},
            )

        self.assertEqual(int(plan.loc[0, "buy_order_qty"]), 100)
        self.assertAlmostEqual(float(plan.loc[0, "estimated_order_fee"]), 20.04, places=6)
        self.assertAlmostEqual(float(summary["estimated_order_fee"]), 20.04, places=6)
        self.assertEqual(summary["execution_model"]["name"], "conservative_v1")

    def test_paper_plan_uses_budget_cap_and_cash_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = SyncConfig(
                scores_path=Path(tmp) / "scores.parquet",
                state_dir=Path(tmp),
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.05,
                budget_total=50_000.0,
                max_buy_order_qty=10_000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )
            latest_scores = pd.DataFrame(
                [
                    {
                        "date": "2026-05-19",
                        "rank": 1,
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "Ping An Bank",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                        "amount": 100_000_000.0,
                        "pct_chg": 0.0,
                        "change": 0.0,
                    }
                ]
            )

            _plan, summary = build_plan(
                config,
                latest_scores=latest_scores,
                positions={},
                balance_metrics={"power": 1_000_000.0, "cash": 1_000_000.0, "total_assets": 1_000_000.0},
            )

        self.assertEqual(summary["total_capital"], 50_000.0)
        self.assertEqual(summary["investable_capital"], 47_500.0)
        self.assertEqual(int(_plan.loc[0, "buy_order_qty"]), 4700)
        self.assertAlmostEqual(float(_plan.loc[0, "estimated_order_notional"]), 47_000.0, places=6)

    def test_paper_latest_scores_filters_st_before_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            scores_path = quant_dir / "models" / "scores.parquet"
            scores_path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"date": "2026-05-20", "code": "688496", "exchange": "SH", "name": "*ST清越", "industry": "Display", "score": 0.99, "close": 1.5},
                    {"date": "2026-05-20", "code": "000001", "exchange": "SZ", "name": "平安银行", "industry": "Bank", "score": 0.8, "close": 10.0},
                ]
            ).to_parquet(scores_path, index=False)
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=quant_dir / "paper_trading",
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=None,
                max_buy_order_qty=1000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )

            latest_scores, _signal_date, _signature = load_latest_scores(config)

        self.assertEqual(latest_scores["code"].tolist(), ["000001"])

    def test_paper_plan_ignores_existing_st_positions_for_manual_handling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            scores_path = quant_dir / "models" / "scores.parquet"
            scores_path.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"code": "688496", "name": "*ST清越"},
                    {"code": "600000", "name": "浦发银行"},
                ]
            ).to_parquet(quant_dir / "stock_list.parquet", index=False)
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=quant_dir / "paper_trading",
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=None,
                max_buy_order_qty=1000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )
            latest_scores = pd.DataFrame(
                [
                    {
                        "date": "2026-05-20",
                        "rank": 1,
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "平安银行",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                    }
                ]
            )

            plan, summary = build_plan(
                config,
                latest_scores=latest_scores,
                positions={
                    "688496": {"symbol": "688496", "quantity": 100, "last_price": 1.5, "market_value": 150.0},
                    "600000": {"symbol": "600000", "quantity": 100, "last_price": 10.0, "market_value": 1000.0},
                },
                balance_metrics={"power": 2000.0, "cash": 2000.0, "total_assets": 3000.0},
            )

        self.assertNotIn("688496", plan["code"].tolist())
        self.assertIn("600000", plan["code"].tolist())
        self.assertEqual(summary["manual_st_positions_ignored"], ["688496"])

    def test_paper_plan_ignores_unmanaged_positions_when_db_managed_positions_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            scores_path = quant_dir / "models" / "scores.parquet"
            scores_path.parent.mkdir(parents=True)
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=quant_dir / "paper_trading",
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=10_000.0,
                max_buy_order_qty=1000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )
            latest_scores = pd.DataFrame(
                [
                    {
                        "date": "2026-05-20",
                        "rank": 1,
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "平安银行",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                    }
                ]
            )

            plan, summary = build_plan(
                config,
                latest_scores=latest_scores,
                positions={
                    "600000": {"symbol": "600000", "quantity": 100, "last_price": 10.0, "market_value": 1000.0},
                },
                balance_metrics={"power": 20_000.0, "cash": 20_000.0, "total_assets": 21_000.0},
                managed_positions={},
            )

        self.assertNotIn("600000", plan["code"].tolist())
        self.assertEqual(summary["managed_position_source"], "paper_managed_positions")
        self.assertEqual(summary["manual_position_symbols_ignored"], ["600000"])

    def test_paper_plan_only_sells_db_managed_quantity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            quant_dir = Path(tmp) / "quant_data"
            scores_path = quant_dir / "models" / "scores.parquet"
            scores_path.parent.mkdir(parents=True)
            config = SyncConfig(
                scores_path=scores_path,
                state_dir=quant_dir / "paper_trading",
                gateway_base_url="http://127.0.0.1:8080",
                market="CN",
                agent_id="agent",
                agent_key="key",
                agent_id_header="X-Agent-Id",
                agent_key_header="X-Agent-Key",
                account_id=None,
                top_k=1,
                min_score=0.5,
                lot_size=100,
                cash_buffer_pct=0.0,
                budget_total=10_000.0,
                max_buy_order_qty=1000,
                max_sell_order_qty=1000,
                cancel_open_orders=True,
                sync_existing_orders=True,
                force=False,
                dry_run=True,
            )
            latest_scores = pd.DataFrame(
                [
                    {
                        "date": "2026-05-20",
                        "rank": 1,
                        "code": "000001",
                        "exchange": "SZ",
                        "name": "平安银行",
                        "industry": "Bank",
                        "score": 0.9,
                        "close": 10.0,
                    }
                ]
            )

            plan, _summary = build_plan(
                config,
                latest_scores=latest_scores,
                positions={
                    "600000": {"symbol": "600000", "quantity": 300, "last_price": 10.0, "market_value": 3000.0},
                },
                balance_metrics={"power": 20_000.0, "cash": 20_000.0, "total_assets": 23_000.0},
                managed_positions={"600000": {"symbol": "600000", "quantity": 100, "avg_cost": 9.0}},
            )

        exit_row = plan[plan["code"].eq("600000")].iloc[0]
        self.assertEqual(int(exit_row["current_qty"]), 100)
        self.assertEqual(int(exit_row["sell_order_qty"]), 100)
        self.assertEqual(exit_row["reason"], "exit_non_target")

    def test_sina_quote_parser_uses_latest_price_field(self) -> None:
        payload = 'var hq_str_sz000001="平安银行,10.860,10.860,10.770,10.880,10.760,10.770,10.780,74763214";'

        self.assertEqual(sina_quote_code("000001", "SZ"), "sz000001")
        self.assertEqual(sina_quote_code("600519", "SH"), "sh600519")
        self.assertAlmostEqual(parse_sina_quote_price(payload, "sz000001"), 10.77, places=6)

    def test_paper_execution_uses_sina_realtime_price_for_order_price(self) -> None:
        class OrderClient:
            def __init__(self) -> None:
                self.orders: list[dict[str, object]] = []

            def place_order(self, **kwargs: object) -> dict[str, object]:
                self.orders.append(dict(kwargs))
                return {"order_id": "order-1", "order_status": "SUBMITTED"}

        client = OrderClient()
        config = SimpleNamespace(
            cancel_open_orders=False,
            max_buy_order_qty=1000,
            max_sell_order_qty=1000,
            execution_model=DEFAULT_EXECUTION_MODEL,
        )
        plan = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "score": 0.9,
                    "code": "000001",
                    "close": 10.0,
                    "amount": 100_000_000.0,
                    "previous_close": 10.0,
                    "buy_limit_price": 10.0,
                    "sell_limit_price": 10.0,
                    "buy_order_qty": 100,
                    "sell_order_qty": 0,
                }
            ]
        )

        with mock.patch("paper_trade_futu.is_active_trading_hours", return_value=True):
            with mock.patch("paper_trade_futu.get_sina_latest_price", return_value=10.77) as get_price:
                result = execute_plan(client, config, plan=plan, signal_date="2026-05-20", active_orders=[])

        self.assertEqual(len(result["placed_orders"]), 1)
        self.assertEqual(result["skipped_orders"], [])
        get_price.assert_called_once_with("000001", None)
        self.assertEqual(client.orders[0]["symbol"], "000001")
        self.assertEqual(client.orders[0]["price"], 10.88)

    def test_paper_execution_uses_separate_buy_sell_quantity_caps(self) -> None:
        class OrderClient:
            def __init__(self) -> None:
                self.orders: list[dict[str, object]] = []

            def place_order(self, **kwargs: object) -> dict[str, object]:
                self.orders.append(dict(kwargs))
                return {"order_id": f"order-{len(self.orders)}", "order_status": "SUBMITTED"}

        client = OrderClient()
        config = SimpleNamespace(
            cancel_open_orders=False,
            max_buy_order_qty=100,
            max_sell_order_qty=9999999999,
            execution_model=DEFAULT_EXECUTION_MODEL,
        )
        plan = pd.DataFrame(
            [
                {"rank": None, "score": None, "code": "605288", "buy_order_qty": 0, "sell_order_qty": 300, "previous_close": 10.0, "amount": 100_000_000.0},
                {"rank": 1, "score": 0.9, "code": "000001", "buy_order_qty": 500, "sell_order_qty": 0, "previous_close": 10.0, "amount": 100_000_000.0},
            ]
        )

        with mock.patch("paper_trade_futu.is_active_trading_hours", return_value=True):
            with mock.patch("paper_trade_futu.get_sina_latest_price", return_value=10.0):
                result = execute_plan(client, config, plan=plan, signal_date="2026-05-20", active_orders=[])

        self.assertEqual(result["skipped_orders"], [])
        self.assertEqual([(row["side"], row["symbol"], row["quantity"]) for row in client.orders], [
            ("SELL", "605288", 300),
            ("BUY", "000001", 100),
        ])

    def test_paper_execution_zero_buy_cap_blocks_buy_orders(self) -> None:
        class OrderClient:
            def __init__(self) -> None:
                self.orders: list[dict[str, object]] = []

            def place_order(self, **kwargs: object) -> dict[str, object]:
                self.orders.append(dict(kwargs))
                return {"order_id": f"order-{len(self.orders)}", "order_status": "SUBMITTED"}

        client = OrderClient()
        config = SimpleNamespace(
            cancel_open_orders=False,
            max_buy_order_qty=0,
            max_sell_order_qty=1000,
            execution_model=DEFAULT_EXECUTION_MODEL,
        )
        plan = pd.DataFrame(
            [
                {"rank": 1, "score": 0.9, "code": "000001", "buy_order_qty": 500, "sell_order_qty": 0, "previous_close": 10.0, "amount": 100_000_000.0},
            ]
        )

        with mock.patch("paper_trade_futu.is_active_trading_hours", return_value=True):
            with mock.patch("paper_trade_futu.get_sina_latest_price", return_value=10.0):
                result = execute_plan(client, config, plan=plan, signal_date="2026-05-20", active_orders=[])

        self.assertEqual(result["placed_orders"], [])
        self.assertEqual(client.orders, [])

    def test_backtest_rebalance_fees_count_buy_and_sell_orders(self) -> None:
        fees = estimate_rebalance_fees(
            previous_symbols={"000001", "000002"},
            next_symbols={"000002", "000003"},
            portfolio_value=10_000.0,
            fee_model=DEFAULT_FEE_MODEL,
        )

        self.assertEqual(fees["buy_count"], 1)
        self.assertEqual(fees["sell_count"], 1)
        self.assertAlmostEqual(float(fees["buy_fee"]), transaction_fee("BUY", 5_000.0), places=6)
        self.assertAlmostEqual(float(fees["sell_fee"]), transaction_fee("SELL", 5_000.0), places=6)

    def test_model_overview_returns_real_profile_equity_curve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            short_3d_run = backtests_dir / "runs" / "20260518T090118Z__short_3d"
            models_dir.mkdir(parents=True)
            short_3d_run.mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                json.dumps({"profile_name": "short_5d"}),
                encoding="utf-8",
            )
            (short_3d_run / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260518T090118Z__short_3d",
                        "profile_name": "short_3d",
                        "profile_label": "3D Short",
                        "portfolio_total_return": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {"rebalance_date": "2026-05-18", "portfolio_return": 0.1, "equity": 1.1, "num_picks": 5},
                    {"rebalance_date": "2026-05-19", "portfolio_return": 0.2, "equity": 1.32, "num_picks": 5},
                ]
            ).to_parquet(short_3d_run / "equity_curve.parquet", index=False)
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview(profile_name="short_3d")

        self.assertEqual(overview["backtest_summary"]["profile_name"], "short_3d")
        self.assertEqual([row["equity"] for row in overview["backtest_equity_curve"]], [1.1, 1.32])

    def test_model_overview_marks_cost_adjusted_backtest_as_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            short_3d_run = backtests_dir / "runs" / "20260520T095341Z__short_3d"
            models_dir.mkdir(parents=True)
            short_3d_run.mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                json.dumps({"profile_name": "short_3d"}),
                encoding="utf-8",
            )
            (short_3d_run / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260520T095341Z__short_3d",
                        "profile_name": "short_3d",
                        "profile_label": "3D Short",
                        "method_version": "purged_label_horizon_costs_v2",
                        "portfolio_total_return": 59.5,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [{"rebalance_date": "2026-05-20", "portfolio_return": 0.1, "equity": 1.1, "num_picks": 5}]
            ).to_parquet(short_3d_run / "equity_curve.parquet", index=False)
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview(profile_name="short_3d")

        self.assertFalse(overview["backtest_summary"]["is_trustworthy"])
        self.assertFalse(overview["backtest_summary"]["is_realistic_execution"])
        self.assertEqual(overview["backtest_summary"]["method_label"], "Research Only")
        self.assertIn("trust_warning", overview["backtest_summary"])

    def test_model_overview_marks_realistic_backtest_as_execution_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            short_3d_run = backtests_dir / "runs" / "20260603T000000Z__short_3d"
            models_dir.mkdir(parents=True)
            short_3d_run.mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                json.dumps({"profile_name": "short_3d"}),
                encoding="utf-8",
            )
            (short_3d_run / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "20260603T000000Z__short_3d",
                        "profile_name": "short_3d",
                        "profile_label": "3D Short",
                        "method_version": "realistic_execution_v1",
                        "portfolio_total_return": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [{"rebalance_date": "2026-05-20", "portfolio_return": 0.1, "equity": 1.1, "num_picks": 5}]
            ).to_parquet(short_3d_run / "equity_curve.parquet", index=False)
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview(profile_name="short_3d")

        self.assertTrue(overview["backtest_summary"]["is_trustworthy"])
        self.assertTrue(overview["backtest_summary"]["is_realistic_execution"])
        self.assertEqual(overview["backtest_summary"]["method_label"], "Realistic Execution")
        self.assertNotIn("trust_warning", overview["backtest_summary"])

    def test_backtest_metric_helpers_report_drawdown_and_annualized_return(self) -> None:
        equity = pd.Series([1.0, 1.2, 0.9, 1.5])
        dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2026-01-01"]))

        self.assertAlmostEqual(max_drawdown(equity), -0.25)
        self.assertGreater(annualized_return(equity, dates), 0.45)

    def test_backtest_training_split_purges_label_horizon(self) -> None:
        dates = pd.Index(pd.to_datetime(["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]))

        cutoff = training_end_for_rebalance(dates, dates[4], label_horizon=2)

        self.assertEqual(pd.Timestamp(cutoff).date().isoformat(), "2026-05-13")

    def test_model_overview_does_not_mix_training_and_backtest_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            models_dir.mkdir(parents=True)
            (backtests_dir / "runs" / "latest_short_3d").mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                '{"profile_name":"short_5d","metrics":{"auc":0.59}}\n',
                encoding="utf-8",
            )
            short_3d_summary = (
                '{"profile_name":"short_3d","profile_label":"3D Short",'
                '"portfolio_total_return":82.21915197894047}\n'
            )
            (backtests_dir / "summary.json").write_text(short_3d_summary, encoding="utf-8")
            (backtests_dir / "runs" / "latest_short_3d" / "summary.json").write_text(short_3d_summary, encoding="utf-8")
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview()

        self.assertEqual(overview["current_profile"], "short_5d")
        self.assertEqual(overview["backtest_summary"], {})
        self.assertEqual(overview["backtest_runs"][0]["profile_name"], "short_3d")

    def test_model_overview_selected_profile_syncs_all_model_panels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            profile_model_dir = root / "quant_data" / "model_profiles" / "short_3d" / "models"
            run_dir = root / "run"
            models_dir.mkdir(parents=True)
            profile_model_dir.mkdir(parents=True)
            (backtests_dir / "runs" / "latest_short_3d").mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                '{"profile_name":"short_5d","metrics":{"auc":0.59}}\n',
                encoding="utf-8",
            )
            (profile_model_dir / "training_metadata.json").write_text(
                '{"profile_name":"short_3d","metrics":{"auc":0.61},"train_rows":123}\n',
                encoding="utf-8",
            )
            (profile_model_dir / "feature_importance.csv").write_text(
                "feature,importance_gain,importance_split\npct_chg,10,2\n",
                encoding="utf-8",
            )
            short_3d_summary = (
                '{"profile_name":"short_3d","profile_label":"3D Short",'
                '"portfolio_total_return":82.21915197894047}\n'
            )
            (backtests_dir / "runs" / "latest_short_3d" / "summary.json").write_text(short_3d_summary, encoding="utf-8")
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview(profile_name="short_3d")

        self.assertEqual(overview["current_profile"], "short_3d")
        self.assertEqual(overview["training_metadata"]["profile_name"], "short_3d")
        self.assertEqual(overview["top_features"][0]["feature"], "pct_chg")
        self.assertEqual(overview["backtest_summary"]["profile_name"], "short_3d")

    def test_latest_picks_can_read_profile_specific_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            profile_model_dir = root / "quant_data" / "model_profiles" / "short_3d" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            profile_model_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)
            backtests_dir.mkdir(parents=True)
            run_dir.mkdir()
            pd.DataFrame(
                [
                    {"date": "2026-05-20", "code": "000001", "name": "A", "industry": "Bank", "score": 0.8, "close": 10.0},
                    {"date": "2026-05-20", "code": "000002", "name": "B", "industry": "Tech", "score": 0.9, "close": 20.0},
                ]
            ).to_parquet(profile_model_dir / "inference_scores_latest.parquet", index=False)
            settings = SimpleNamespace(
                models_dir=models_dir,
                backtests_dir=backtests_dir,
                run_dir=run_dir,
                quant_dir=root / "quant_data",
                stock_list_path=root / "missing_stock_list.parquet",
                stock_registry_path=root / "missing_stock_registry.parquet",
            )

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with mock.patch.object(
                        model_service,
                        "_scores_path_for_profile",
                        return_value=("short_3d", profile_model_dir / "inference_scores_latest.parquet"),
                    ):
                        picks = model_service.get_latest_picks(limit=1, profile_name="short_3d")

        self.assertEqual(picks["profile_name"], "short_3d")
        self.assertEqual(picks["picks"][0]["code"], "000002")

    def test_activate_model_for_paper_uses_atomic_registry_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            profile_model_dir = root / "quant_data" / "model_profiles" / "short_3d" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            profile_model_dir.mkdir(parents=True)
            models_dir.mkdir(parents=True)
            backtests_dir.mkdir(parents=True)
            run_dir.mkdir()
            (run_dir / "model_profiles.json").write_text(
                json.dumps(
                    {
                        "default_profile": "short_5d",
                        "profiles": [
                            {"name": "short_5d", "label": "5D Short"},
                            {"name": "short_3d", "label": "3D Short"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (profile_model_dir / "training_metadata.json").write_text('{"profile_name":"short_3d"}\n', encoding="utf-8")
            (profile_model_dir / "inference_scores_latest.parquet").write_bytes(b"score-bytes")
            (profile_model_dir / "feature_importance.csv").write_text("feature,importance_gain\nx,1\n", encoding="utf-8")
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with mock.patch.object(
                        model_service,
                        "activate_model",
                        return_value={
                            "model_version": "cn-short-3d-v1",
                            "market": "CN",
                            "paper_enabled": True,
                            "revision": 2,
                            "activated_at": "2026-08-13T00:00:00+00:00",
                        },
                    ) as activate:
                        result = model_service.activate_model_for_paper("short_3d")

            self.assertEqual(result["profile_name"], "short_3d")
            self.assertEqual(result["model_version"], "cn-short-3d-v1")
            self.assertFalse((models_dir / "training_metadata.json").exists())
            self.assertNotIn("active_profile", json.loads((run_dir / "model_profiles.json").read_text(encoding="utf-8")))
            activate.assert_called_once_with(
                market="CN",
                profile="short_3d",
                paper_enabled=True,
                actor="panel_admin",
                reason="Activated from the control panel.",
            )

    def test_model_overview_does_not_default_unprofiled_backtest_to_current_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models_dir = root / "quant_data" / "models"
            backtests_dir = root / "quant_data" / "backtests"
            run_dir = root / "run"
            models_dir.mkdir(parents=True)
            backtests_dir.mkdir(parents=True)
            run_dir.mkdir()
            (models_dir / "training_metadata.json").write_text(
                '{"profile_name":"short_5d","metrics":{"auc":0.59}}\n',
                encoding="utf-8",
            )
            (backtests_dir / "summary.json").write_text(
                '{"portfolio_total_return":82.21915197894047}\n',
                encoding="utf-8",
            )
            settings = SimpleNamespace(models_dir=models_dir, backtests_dir=backtests_dir, run_dir=run_dir, quant_dir=root / "quant_data")

            with mock.patch.object(model_service, "get_settings", return_value=settings):
                with mock.patch.object(model_profiles_service, "get_settings", return_value=settings):
                    with self._mock_model_registry(settings):
                        overview = model_service.get_model_overview()

        self.assertEqual(overview["current_profile"], "short_5d")
        self.assertEqual(overview["backtest_summary"], {})

    def test_legacy_provider_logs_are_translated_before_display(self) -> None:
        line = "\u8bf7\u6c42\u5931\u8d25\uff0c1.0 \u79d2\u540e\u91cd\u8bd5 (1/3): timeout"

        self.assertEqual(translate_log_line(line), "Request failed, retrying in 1.0s (1/3): timeout")

    def test_public_ui_sources_are_english_only(self) -> None:
        checked_roots = [ROOT / "apps" / "web" / "app", ROOT / "apps" / "web" / "lib"]
        offenders: list[str] = []
        for root in checked_roots:
            for path in root.rglob("*.ts*"):
                text = path.read_text(encoding="utf-8")
                if any("\u4e00" <= char <= "\u9fff" for char in text):
                    offenders.append(str(path.relative_to(ROOT)))

        self.assertEqual(offenders, [])

    def test_results_doc_uses_aggregate_metrics_only(self) -> None:
        results_doc = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")

        self.assertIn("Production Research Results", results_doc)
        self.assertIn("Validation AUC", results_doc)
        self.assertIn("Walk-Forward OOS Backtest", results_doc)
        self.assertNotIn("FUTU_GATEWAY", results_doc)
        self.assertNotIn("account_id", results_doc.lower())


if __name__ == "__main__":
    unittest.main()
