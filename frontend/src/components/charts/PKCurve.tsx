import { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { exportCSV } from "../../lib/utils";

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

export default function PKCurve({ datasets, mecLine, mtcLine }: Props) {
  const [logScale, setLogScale] = useState(false);

  const chartData = useMemo(() => {
    if (datasets.length === 0) return [];
    // Merge all datasets by time index (assume first dataset is primary)
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

  const handleExportCSV = () => {
    if (datasets.length === 0) return;
    const ds = datasets[0];
    exportCSV(ds.time, ds.conc, "pk_curve.csv");
  };

  const handleCopy = () => {
    if (datasets.length === 0) return;
    const ds = datasets[0];
    const header = "time_h\tcp_mg_L";
    const rows = ds.time.map((t, i) => `${t}\t${ds.conc[i]}`);
    navigator.clipboard.writeText([header, ...rows].join("\n")).catch(() => {});
  };

  if (datasets.length === 0) return null;

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-[var(--text)]">
          Concentration-Time Profile
        </h3>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setLogScale((p) => !p)}
            className="text-xs px-2 py-1 rounded border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--border)] transition-colors"
          >
            {logScale ? "Linear" : "Semi-log"}
          </button>
          <button
            onClick={handleExportCSV}
            className="text-xs px-2 py-1 rounded border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--border)] transition-colors"
          >
            CSV
          </button>
          <button
            onClick={handleCopy}
            className="text-xs px-2 py-1 rounded border border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--border)] transition-colors"
          >
            Copy
          </button>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="time"
            stroke="#737373"
            tick={{ fontSize: 11, fill: "#a3a3a3" }}
            label={{ value: "Time (h)", position: "insideBottom", offset: -10, fill: "#a3a3a3", fontSize: 12 }}
          />
          <YAxis
            stroke="#737373"
            tick={{ fontSize: 11, fill: "#a3a3a3" }}
            scale={logScale ? "log" : "auto"}
            domain={logScale ? ["auto", "auto"] : [0, "auto"]}
            allowDataOverflow={logScale}
            label={{ value: "Cp (mg/L)", angle: -90, position: "insideLeft", offset: 5, fill: "#a3a3a3", fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{ backgroundColor: "#1a1a1a", border: "1px solid #333", borderRadius: 6, fontSize: 12 }}
            labelStyle={{ color: "#a3a3a3" }}
            itemStyle={{ color: "#fafafa" }}
            labelFormatter={(v) => `${v} h`}
          />
          {mecLine != null && (
            <ReferenceLine y={mecLine} stroke="#22c55e" strokeDasharray="5 3" label={{ value: "MEC", fill: "#22c55e", fontSize: 11 }} />
          )}
          {mtcLine != null && (
            <ReferenceLine y={mtcLine} stroke="#ef4444" strokeDasharray="5 3" label={{ value: "MTC", fill: "#ef4444", fontSize: 11 }} />
          )}
          {datasets.map((ds) => (
            <Line
              key={ds.label}
              type="monotone"
              dataKey={ds.label}
              stroke={ds.color}
              strokeWidth={2}
              dot={false}
              name={ds.label}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
