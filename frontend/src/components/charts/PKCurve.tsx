import { useState, useMemo, useCallback } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { exportCSV, formatNum } from "../../lib/utils";

interface Dataset {
  time: number[];
  conc: number[];
  label: string;
  color: string;
}

interface Props {
  datasets: Dataset[];
  mecLine?: number;
  mtcLine?: number;
}

export type { Dataset };

/* Custom tooltip */
function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value: number; color: string }>;
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[#1a1a1a] px-3 py-2 shadow-xl">
      <div className="text-xs text-[var(--text-muted)] mb-1">{formatNum(label ?? 0)} h</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-sm">
          <span
            className="w-2 h-2 rounded-full shrink-0"
            style={{ background: p.color }}
          />
          <span className="text-[var(--text-muted)]">{p.name}:</span>
          <span className="font-medium text-[var(--text)] tabular-nums">
            {formatNum(p.value, 4)} mg/L
          </span>
        </div>
      ))}
    </div>
  );
}

export default function PKCurve({ datasets, mecLine, mtcLine }: Props) {
  const [logScale, setLogScale] = useState(false);

  const chartData = useMemo(() => {
    if (datasets.length === 0) return [];
    const primary = datasets[0];
    return primary.time.map((t, i) => {
      const point: Record<string, number> = { time: t };
      for (const ds of datasets) {
        const val = ds.conc[i] ?? 0;
        point[ds.label] = logScale ? Math.max(val, 1e-6) : val;
      }
      return point;
    });
  }, [datasets, logScale]);

  const handleExportCSV = useCallback(() => {
    if (datasets.length === 0) return;
    const ds = datasets[0];
    exportCSV(ds.time, ds.conc, "pk_curve.csv");
  }, [datasets]);

  const handleCopy = useCallback(() => {
    if (datasets.length === 0) return;
    const ds = datasets[0];
    const header = "time_h\tcp_mg_L";
    const rows = ds.time.map((t, i) => `${t}\t${ds.conc[i]}`);
    navigator.clipboard.writeText([header, ...rows].join("\n")).catch(() => {});
  }, [datasets]);

  if (datasets.length === 0) return null;

  const isSingle = datasets.length === 1;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-[var(--text)]">
          Concentration-Time Profile
        </h3>
        <div className="flex items-center gap-1.5">
          {["Linear", "Semi-log"].map((label, idx) => {
            const isActive = idx === 0 ? !logScale : logScale;
            return (
              <button
                key={label}
                onClick={() => setLogScale(idx === 1)}
                className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
                  isActive
                    ? "bg-blue-500/15 text-blue-400 font-medium"
                    : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5"
                }`}
              >
                {label}
              </button>
            );
          })}
          <div className="w-px h-4 bg-[var(--border)] mx-1" />
          <button
            onClick={handleCopy}
            className="text-xs px-2 py-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5 transition-colors"
            title="Copy to clipboard"
          >
            Copy
          </button>
          <button
            onClick={handleExportCSV}
            className="text-xs px-2 py-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5 transition-colors"
            title="Download CSV"
          >
            CSV
          </button>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <AreaChart
          data={chartData}
          margin={{ top: 5, right: 20, bottom: 25, left: 15 }}
        >
          <defs>
            {datasets.map((ds) => (
              <linearGradient
                key={`grad-${ds.label}`}
                id={`gradient-${ds.label.replace(/\s/g, "-")}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop offset="0%" stopColor={ds.color} stopOpacity={0.25} />
                <stop offset="95%" stopColor={ds.color} stopOpacity={0.02} />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="time"
            stroke="transparent"
            tick={{ fontSize: 11, fill: "#737373" }}
            tickLine={false}
            axisLine={false}
            label={{
              value: "Time (h)",
              position: "insideBottom",
              offset: -15,
              fill: "#737373",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="transparent"
            tick={{ fontSize: 11, fill: "#737373" }}
            tickLine={false}
            axisLine={false}
            scale={logScale ? "log" : "auto"}
            domain={logScale ? ["auto", "auto"] : [0, "auto"]}
            allowDataOverflow={logScale}
            label={{
              value: "Cp (mg/L)",
              angle: -90,
              position: "insideLeft",
              offset: 0,
              fill: "#737373",
              fontSize: 11,
            }}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{
              stroke: "rgba(255,255,255,0.15)",
              strokeWidth: 1,
              strokeDasharray: "4 4",
            }}
          />
          {mecLine != null && (
            <ReferenceLine
              y={mecLine}
              stroke="#22c55e"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: "MEC",
                fill: "#22c55e",
                fontSize: 10,
                position: "right",
              }}
            />
          )}
          {mtcLine != null && (
            <ReferenceLine
              y={mtcLine}
              stroke="#ef4444"
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: "MTC",
                fill: "#ef4444",
                fontSize: 10,
                position: "right",
              }}
            />
          )}
          {datasets.map((ds) => (
            <Area
              key={ds.label}
              type="monotone"
              dataKey={ds.label}
              stroke={ds.color}
              strokeWidth={2}
              fill={
                isSingle
                  ? `url(#gradient-${ds.label.replace(/\s/g, "-")})`
                  : "transparent"
              }
              dot={false}
              name={ds.label}
              animationDuration={600}
              animationEasing="ease-out"
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>

      {/* Legend for multi-dataset */}
      {datasets.length > 1 && (
        <div className="flex items-center gap-4 mt-3 ml-16">
          {datasets.map((ds) => (
            <div key={ds.label} className="flex items-center gap-1.5 text-xs">
              <span
                className="w-3 h-0.5 rounded-full inline-block"
                style={{ background: ds.color }}
              />
              <span className="text-[var(--text-muted)]">{ds.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
