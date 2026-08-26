import { MetricCard, Panel } from "@/components/cards";
import { Shell } from "@/components/shell";
import { DataTable } from "@/components/table";
import { getPaperHoldings, getPaperTargets, getPortfolioOverview } from "@/lib/api";
import { requireAuth } from "@/lib/auth";
import { formatDate, formatNumber } from "@/lib/format";

export const dynamic = "force-dynamic";

type Row = Record<string, unknown>;

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function symbolFrom(row: Row) {
  const raw = String(row.symbol ?? row.code ?? "").trim().toUpperCase();
  if (!raw) return "";
  if (raw.includes(".")) {
    const numeric = raw.split(".").find((part) => /^\d+$/.test(part));
    if (numeric) return numeric.padStart(6, "0");
  }
  return /^\d+$/.test(raw) ? raw.padStart(6, "0") : raw;
}

export default async function CnPortfolioPage() {
  const user = await requireAuth();
  const [overview, holdings, targets] = await Promise.all([
    getPortfolioOverview(),
    getPaperHoldings(120, 0, 3200).catch((error) => ({
      summary: {},
      balance: [],
      positions_rows: 0,
      positions: [],
      orders_rows: 0,
      orders: [],
      error: error instanceof Error ? error.message : "Portfolio positions are temporarily unavailable."
    })),
    getPaperTargets(60).catch(() => ({ rows: 0, targets: [] }))
  ]);

  const positionRows = (holdings.positions as Row[]).filter((row) => {
    return Math.abs(asNumber(row.quantity ?? row.qty) ?? 0) > 0 || Math.abs(asNumber(row.market_value) ?? 0) > 0;
  });
  const positionMarketValue = positionRows.reduce((total, row) => total + (asNumber(row.market_value) ?? 0), 0);
  const positions = positionRows.map((row) => {
    const marketValue = asNumber(row.market_value);
    const symbol = symbolFrom(row);
    return {
      symbol,
      symbol_href: symbol ? `/paper/stocks/${encodeURIComponent(symbol)}` : undefined,
      name: row.name ?? row.stock_name ?? null,
      quantity: row.quantity ?? row.qty ?? null,
      avg_cost: row.avg_cost ?? row.cost_price ?? null,
      last_price: row.last_price ?? row.price ?? null,
      market_value: marketValue,
      actual_weight_pct: positionMarketValue > 0 && marketValue !== null ? (marketValue / positionMarketValue) * 100 : null,
      unrealized_pnl: row.unrealized_pnl ?? null
    };
  });
  const targetRows = (targets.targets as Row[]).map((row) => {
    const symbol = symbolFrom(row);
    const targetWeight = asNumber(row.target_weight);
    return {
      rank: row.rank ?? null,
      symbol,
      symbol_href: symbol ? `/paper/stocks/${encodeURIComponent(symbol)}` : undefined,
      name: row.name ?? row.stock_name ?? null,
      action: row.action ?? null,
      target_weight_pct: targetWeight === null ? null : targetWeight * 100,
      target_qty: row.target_qty ?? null,
      current_qty: row.current_qty ?? null,
      delta_qty: row.delta_qty ?? null
    };
  });
  const account = overview.account;

  return (
    <Shell
      title="CN Stocks Portfolio"
      subtitle="Positions, performance and validated target weights"
      locale={user.locale}
      username={user.username}
      role={user.role}
      market="CN"
    >
      <section className="product-stage-heading">
        <div><span className="stage-icon">◆</span><div><h1>CN Portfolio</h1><p>Account positions, P&amp;L and the latest strategy targets.</p></div></div>
        <span className="capability-label status-live">Live</span>
      </section>

      {overview.warnings.map((warning) => <p className="banner banner-info" key={warning}>{warning}</p>)}
      {holdings.error ? <p className="banner banner-info">Showing the latest portfolio summary while live positions refresh.</p> : null}

      <section className="metrics-grid compact-metrics">
        <MetricCard label="Account Equity" value={formatNumber(account.total_assets, user.locale, { maximumFractionDigits: 0 })} hint={account.currency || "CNY"} />
        <MetricCard label="Market Value" value={formatNumber(account.market_value, user.locale, { maximumFractionDigits: 0 })} />
        <MetricCard label="Cash" value={formatNumber(account.cash, user.locale, { maximumFractionDigits: 0 })} />
        <MetricCard label="Total P&L" value={formatNumber(account.total_pnl, user.locale, { maximumFractionDigits: 0 })} />
        <MetricCard label="Today P&L" value={formatNumber(account.today_pnl, user.locale, { maximumFractionDigits: 0 })} />
        <MetricCard label="Latest Target" value={formatDate(overview.signals.latest_signal_date, user.locale)} hint={`${formatNumber(targetRows.length, user.locale)} symbols`} />
      </section>

      <Panel title="Current Positions" aside={<span className="pill">{formatNumber(positions.length, user.locale)} holdings</span>}>
        <DataTable
          rows={positions}
          columns={[
            { key: "symbol", label: "Symbol" },
            { key: "name", label: "Company" },
            { key: "quantity", label: "Quantity" },
            { key: "avg_cost", label: "Avg Cost" },
            { key: "last_price", label: "Last Price" },
            { key: "market_value", label: "Market Value" },
            { key: "actual_weight_pct", label: "Weight %" },
            { key: "unrealized_pnl", label: "Unrealized P&L" }
          ]}
          emptyLabel="No current positions."
          locale={user.locale}
          pageSize={25}
        />
      </Panel>

      <Panel title="Strategy Targets" aside={<span className="pill">Latest validated rebalance</span>}>
        <DataTable
          rows={targetRows}
          columns={[
            { key: "rank", label: "Rank" },
            { key: "symbol", label: "Symbol" },
            { key: "name", label: "Company" },
            { key: "action", label: "Action" },
            { key: "target_weight_pct", label: "Target Weight %" },
            { key: "target_qty", label: "Target Qty" },
            { key: "current_qty", label: "Current Qty" },
            { key: "delta_qty", label: "Delta" }
          ]}
          emptyLabel="No validated target basket is currently available."
          locale={user.locale}
          pageSize={25}
        />
      </Panel>
    </Shell>
  );
}
