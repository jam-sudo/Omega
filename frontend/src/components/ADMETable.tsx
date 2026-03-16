import { formatNum } from "../lib/utils";
import { AlertTriangle, CheckCircle, Info } from "lucide-react";

interface Props {
  adme: Record<string, number | string>;
  reference?: Record<string, number | string>;
}

interface PropertyDef {
  key: string;
  label: string;
  unit: string;
  range?: [number, number]; // normal range for bar visualization
  warnLow?: number;
  warnHigh?: number;
  category: "physicochemical" | "absorption" | "metabolism" | "safety";
}

const PROPERTIES: PropertyDef[] = [
  { key: "mw", label: "Molecular Weight", unit: "g/mol", range: [100, 900], warnHigh: 500, category: "physicochemical" },
  { key: "logP", label: "logP", unit: "", range: [-2, 7], warnHigh: 5, category: "physicochemical" },
  { key: "logS", label: "logS", unit: "log mol/L", range: [-7, 0], warnLow: -5, category: "physicochemical" },
  { key: "fup", label: "Fraction Unbound", unit: "", range: [0, 1], warnLow: 0.01, category: "absorption" },
  { key: "rbp", label: "Blood:Plasma Ratio", unit: "", range: [0.5, 3], category: "absorption" },
  { key: "peff", label: "Permeability", unit: "×10⁻⁴ cm/s", range: [0, 10], warnLow: 0.5, category: "absorption" },
  { key: "clint_3a4", label: "CLint CYP3A4", unit: "µL/min/pmol", range: [0, 50], warnHigh: 20, category: "metabolism" },
  { key: "herg_ic50_uM", label: "hERG IC50", unit: "µM", range: [0, 100], warnLow: 10, category: "safety" },
];

const CATEGORY_LABELS: Record<string, string> = {
  physicochemical: "Physicochemical",
  absorption: "Absorption & Distribution",
  metabolism: "Metabolism",
  safety: "Safety",
};

function getStatus(val: number, prop: PropertyDef): "ok" | "warn" | "neutral" {
  if (prop.warnLow !== undefined && val < prop.warnLow) return "warn";
  if (prop.warnHigh !== undefined && val > prop.warnHigh) return "warn";
  return "ok";
}

function StatusIcon({ status }: { status: "ok" | "warn" | "neutral" }) {
  if (status === "warn")
    return <AlertTriangle size={14} className="text-yellow-400 shrink-0" />;
  if (status === "ok")
    return <CheckCircle size={14} className="text-green-400/60 shrink-0" />;
  return <Info size={14} className="text-[var(--text-muted)] shrink-0" />;
}

function RangeBar({ value, range, status }: { value: number; range: [number, number]; status: "ok" | "warn" | "neutral" }) {
  const [lo, hi] = range;
  const pct = Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
  const barColor = status === "warn" ? "bg-yellow-400" : "bg-blue-400";

  return (
    <div className="w-20 h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
      <div
        className={`h-full rounded-full transition-all ${barColor}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export default function ADMETable({ adme, reference }: Props) {
  const categories = [...new Set(PROPERTIES.map((p) => p.category))];

  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <h3 className="text-sm font-medium text-[var(--text)]">ADME Properties</h3>
      </div>

      {categories.map((cat) => {
        const props = PROPERTIES.filter((p) => p.category === cat);
        const hasAny = props.some((p) => adme[p.key] != null);
        if (!hasAny) return null;

        return (
          <div key={cat}>
            <div className="px-4 py-1.5 bg-white/[0.02]">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
                {CATEGORY_LABELS[cat]}
              </span>
            </div>
            {props.map((prop) => {
              const val = adme[prop.key];
              if (val == null) return null;
              const numVal = typeof val === "number" ? val : parseFloat(String(val));
              const status = isNaN(numVal) ? "neutral" as const : getStatus(numVal, prop);
              const lo = adme[`${prop.key}_lo`];
              const hi = adme[`${prop.key}_hi`];

              return (
                <div
                  key={prop.key}
                  className="flex items-center gap-3 px-4 py-2 border-t border-[var(--border)] hover:bg-white/[0.02] transition-colors"
                >
                  <StatusIcon status={status} />
                  <div className="flex-1 min-w-0">
                    <span className="text-sm text-[var(--text)]">{prop.label}</span>
                    {prop.unit && (
                      <span className="text-[var(--text-muted)] text-xs ml-1">({prop.unit})</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {prop.range && !isNaN(numVal) && (
                      <RangeBar value={numVal} range={prop.range} status={status} />
                    )}
                    <span className="text-sm font-medium tabular-nums text-[var(--text)] w-16 text-right">
                      {typeof val === "number" ? formatNum(val) : val}
                    </span>
                    {reference && reference[prop.key] != null && (
                      <span className="text-xs text-orange-400/80 tabular-nums w-16 text-right">
                        {formatNum(Number(reference[prop.key]))}
                      </span>
                    )}
                    <span className="text-xs tabular-nums text-[var(--text-muted)] w-24 text-right">
                      {typeof lo === "number" && typeof hi === "number"
                        ? `${formatNum(lo)} – ${formatNum(hi)}`
                        : ""}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
