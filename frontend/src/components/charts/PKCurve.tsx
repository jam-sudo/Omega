import { useState, useMemo, useCallback, useRef } from "react";
import {
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Scatter,
  ErrorBar,
  ComposedChart,
  Brush,
} from "recharts";
import { exportCSV, formatNum } from "../../lib/utils";

interface Dataset {
  time: number[];
  conc: number[];
  label: string;
  color: string;
  sd?: number[];
  isReference?: boolean;
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
    <div className="rounded-lg border border-[var(--border)] bg-[#18181b] px-3 py-2 shadow-xl">
      <div className="text-xs text-[var(--text-muted)] mb-1">{formatNum(label ?? 0)} h</div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: p.color }} />
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

  // Zoom state: user drags on chart to select X range
  const [zoomLeft, setZoomLeft] = useState<number | null>(null);
  const [zoomRight, setZoomRight] = useState<number | null>(null);
  const [xDomain, setXDomain] = useState<[number, number] | null>(null);
  const [yDomain, setYDomain] = useState<[number, number] | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);

  const lineDatasets = useMemo(() => datasets.filter((ds) => !ds.isReference), [datasets]);
  const refDatasets = useMemo(() => datasets.filter((ds) => ds.isReference), [datasets]);

  const chartData = useMemo(() => {
    if (lineDatasets.length === 0) return [];
    const primary = lineDatasets[0];
    return primary.time.map((t, i) => {
      const point: Record<string, number> = { time: t };
      for (const ds of lineDatasets) {
        const val = ds.conc[i] ?? 0;
        point[ds.label] = logScale ? Math.max(val, 1e-6) : val;
      }
      return point;
    });
  }, [lineDatasets, logScale]);

  const refChartData = useMemo(() => {
    return refDatasets.map((ds) =>
      ds.time.map((t, i) => ({
        time: t,
        conc: logScale ? Math.max(ds.conc[i] ?? 0, 1e-6) : (ds.conc[i] ?? 0),
        sdUpper: ds.sd?.[i] ?? 0,
        sdLower: ds.sd?.[i] ?? 0,
      })),
    );
  }, [refDatasets, logScale]);

  // Compute Y domain when X is zoomed
  useMemo(() => {
    if (!xDomain) {
      setYDomain(null);
      return;
    }
    const [xMin, xMax] = xDomain;
    let yMax = 0;
    for (const ds of lineDatasets) {
      for (let i = 0; i < ds.time.length; i++) {
        if (ds.time[i] >= xMin && ds.time[i] <= xMax) {
          yMax = Math.max(yMax, ds.conc[i] ?? 0);
        }
      }
    }
    for (const ds of refDatasets) {
      for (let i = 0; i < ds.time.length; i++) {
        if (ds.time[i] >= xMin && ds.time[i] <= xMax) {
          yMax = Math.max(yMax, (ds.conc[i] ?? 0) + (ds.sd?.[i] ?? 0));
        }
      }
    }
    if (yMax > 0) {
      setYDomain([0, yMax * 1.1]);
    }
  }, [xDomain, lineDatasets, refDatasets]);

  const handleMouseDown = useCallback(
    (e: { activeLabel?: string | number }) => {
      if (e?.activeLabel != null) setZoomLeft(Number(e.activeLabel));
    },
    [],
  );

  const handleMouseMove = useCallback(
    (e: { activeLabel?: string | number }) => {
      if (zoomLeft != null && e?.activeLabel != null) setZoomRight(Number(e.activeLabel));
    },
    [zoomLeft],
  );

  const handleMouseUp = useCallback(() => {
    if (zoomLeft != null && zoomRight != null && zoomLeft !== zoomRight) {
      const left = Math.min(zoomLeft, zoomRight);
      const right = Math.max(zoomLeft, zoomRight);
      setXDomain([left, right]);
    }
    setZoomLeft(null);
    setZoomRight(null);
  }, [zoomLeft, zoomRight]);

  const handleReset = useCallback(() => {
    setXDomain(null);
    setYDomain(null);
  }, []);

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

  const hasReference = refDatasets.length > 0;
  const isSingle = lineDatasets.length === 1 && !hasReference;
  const isZoomed = xDomain != null;

  const computedXDomain: [string | number, string | number] = xDomain || ["dataMin", "dataMax"];
  const computedYDomain: [string | number, string | number] = logScale
    ? ["auto", "auto"]
    : yDomain || [0, "auto"];

  return (
    <div ref={chartRef} className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
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
                    ? "bg-white/[0.08] text-[var(--text)] font-medium"
                    : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5"
                }`}
              >
                {label}
              </button>
            );
          })}
          {isZoomed && (
            <>
              <div className="w-px h-4 bg-[var(--border)] mx-1" />
              <button
                onClick={handleReset}
                className="text-xs px-2 py-1 rounded-md text-orange-400 hover:bg-orange-500/10 transition-colors"
              >
                Reset zoom
              </button>
            </>
          )}
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

      {/* Zoom hint */}
      {!isZoomed && (
        <div className="text-[10px] text-[var(--text-dim)] mb-2 ml-12">
          Drag on chart to zoom · Double-click to reset
        </div>
      )}

      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart
          data={chartData}
          margin={{ top: 5, right: 20, bottom: 25, left: 15 }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onDoubleClick={handleReset}
        >
          <defs>
            {lineDatasets.map((ds) => (
              <linearGradient
                key={`grad-${ds.label}`}
                id={`gradient-${ds.label.replace(/\s/g, "-")}`}
                x1="0" y1="0" x2="0" y2="1"
              >
                <stop offset="0%" stopColor={ds.color} stopOpacity={0.2} />
                <stop offset="95%" stopColor={ds.color} stopOpacity={0.01} />
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
            tick={{ fontSize: 11, fill: "#52525b" }}
            tickLine={false}
            axisLine={false}
            type="number"
            domain={computedXDomain}
            allowDataOverflow={isZoomed}
            label={{
              value: "Time (h)",
              position: "insideBottom",
              offset: -15,
              fill: "#52525b",
              fontSize: 11,
            }}
          />
          <YAxis
            stroke="transparent"
            tick={{ fontSize: 11, fill: "#52525b" }}
            tickLine={false}
            axisLine={false}
            scale={logScale ? "log" : "auto"}
            domain={computedYDomain}
            allowDataOverflow={logScale || isZoomed}
            label={{
              value: "Cp (mg/L)",
              angle: -90,
              position: "insideLeft",
              offset: 0,
              fill: "#52525b",
              fontSize: 11,
            }}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{
              stroke: "rgba(255,255,255,0.1)",
              strokeWidth: 1,
              strokeDasharray: "4 4",
            }}
          />

          {/* Zoom selection overlay */}
          {zoomLeft != null && zoomRight != null && (
            <ReferenceLine x={zoomLeft} stroke="rgba(255,255,255,0.2)" strokeDasharray="2 2" />
          )}
          {zoomLeft != null && zoomRight != null && (
            <ReferenceLine x={zoomRight} stroke="rgba(255,255,255,0.2)" strokeDasharray="2 2" />
          )}

          {mecLine != null && (
            <ReferenceLine
              y={mecLine} stroke="#22c55e" strokeDasharray="6 3" strokeWidth={1.5}
              label={{ value: "MEC", fill: "#22c55e", fontSize: 10, position: "right" }}
            />
          )}
          {mtcLine != null && (
            <ReferenceLine
              y={mtcLine} stroke="#ef4444" strokeDasharray="6 3" strokeWidth={1.5}
              label={{ value: "MTC", fill: "#ef4444", fontSize: 10, position: "right" }}
            />
          )}

          {lineDatasets.map((ds) => (
            <Area
              key={ds.label}
              type="monotone"
              dataKey={ds.label}
              stroke={ds.color}
              strokeWidth={1.5}
              fill={isSingle ? `url(#gradient-${ds.label.replace(/\s/g, "-")})` : "transparent"}
              dot={false}
              name={ds.label}
              animationDuration={400}
              animationEasing="ease-out"
            />
          ))}
          {refDatasets.map((ds, idx) => (
            <Scatter
              key={ds.label}
              name={ds.label}
              data={refChartData[idx]}
              fill={ds.color}
              dataKey="conc"
              shape={(props) => {
                if (props.cx == null || props.cy == null) return null;
                return (
                  <circle
                    cx={props.cx} cy={props.cy} r={3}
                    fill={ds.color} stroke={ds.color}
                    strokeWidth={1} fillOpacity={0.8}
                  />
                );
              }}
            >
              {ds.sd && (
                <ErrorBar dataKey="sdUpper" width={3} strokeWidth={1} stroke={ds.color} direction="y" />
              )}
            </Scatter>
          ))}

          {/* Bottom brush for panning when zoomed */}
          {isZoomed && (
            <Brush
              dataKey="time"
              height={20}
              stroke="rgba(255,255,255,0.1)"
              fill="#18181b"
              travellerWidth={8}
              onChange={(range) => {
                if (range.startIndex != null && range.endIndex != null && chartData.length > 0) {
                  const start = chartData[range.startIndex]?.time;
                  const end = chartData[range.endIndex]?.time;
                  if (start != null && end != null) {
                    setXDomain([start, end]);
                  }
                }
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      {(lineDatasets.length > 1 || hasReference) && (
        <div className="flex items-center gap-4 mt-3 ml-16">
          {lineDatasets.map((ds) => (
            <div key={ds.label} className="flex items-center gap-1.5 text-xs">
              <span className="w-3 h-0.5 rounded-full inline-block" style={{ background: ds.color }} />
              <span className="text-[var(--text-muted)]">{ds.label}</span>
            </div>
          ))}
          {refDatasets.map((ds) => (
            <div key={ds.label} className="flex items-center gap-1.5 text-xs">
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: ds.color }} />
              <span className="text-[var(--text-muted)]">{ds.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
