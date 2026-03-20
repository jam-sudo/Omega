import { formatNum } from "../../lib/utils";

interface CardData {
  label: string;
  value: number;
  unit: string;
  p5?: number;
  p95?: number;
  reference?: number;
  foldError?: number;
  referenceLabel?: string;
  isTime?: boolean;
}

interface Props {
  cards: CardData[];
  confidence?: string;
}

export type { CardData };

const CARD_ACCENTS: Record<string, string> = {
  Cmax: "#3b82f6",
  Tmax: "#06b6d4",
  "t\u00BD": "#f59e0b",
};

const AUC_ACCENT = "#8b5cf6";

function rangePercent(value: number, p5?: number, p95?: number): number {
  if (p5 == null || p95 == null || p95 === p5) return 50;
  return Math.max(5, Math.min(95, ((value - p5) / (p95 - p5)) * 100));
}

function foldErrorColor(fe: number): string {
  if (fe <= 2) return "text-green-400";
  if (fe <= 3) return "text-yellow-400";
  return "text-red-400";
}

export default function PKCards({ cards }: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => {
        const accent = CARD_ACCENTS[c.label] || AUC_ACCENT;
        const barPct = rangePercent(c.value, c.p5, c.p95);

        return (
          <div
            key={c.label}
            className="bg-[#18181b] border border-[#27272a] rounded-lg p-4"
          >
            <div className="text-[10px] text-[#71717a] uppercase tracking-[1.5px] font-medium">
              {c.label}
            </div>
            <div className="flex items-baseline gap-1.5 mt-1.5">
              {c.isTime ? (
                <span className="text-[28px] font-light text-[#fafafa] tracking-tight font-mono">
                  {c.unit}
                </span>
              ) : (
                <>
                  <span className="text-[32px] font-light text-[#fafafa] tabular-nums tracking-tight font-mono">
                    {formatNum(c.value)}
                  </span>
                  <span className="text-xs text-[#52525b]">{c.unit}</span>
                </>
              )}
            </div>
            {/* Range bar */}
            <div className="w-full h-[3px] bg-[#27272a] rounded-full mt-3 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${barPct}%`, background: accent }}
              />
            </div>
            {/* Reference */}
            {c.reference != null && c.foldError != null && (
              <div className="flex justify-between mt-1.5">
                <span className="text-[10px] text-[#52525b]">
                  {c.referenceLabel || "ref"} {formatNum(c.reference)}
                </span>
                <span className={`text-[10px] ${foldErrorColor(c.foldError)}`}>
                  {formatNum(c.foldError)}&times;
                </span>
              </div>
            )}
            {/* CI range (no ref) */}
            {c.reference == null && c.p5 != null && c.p95 != null && (
              <div className="mt-1.5">
                <span className="text-[10px] text-[#52525b] tabular-nums">
                  90% CI: {formatNum(c.p5)} – {formatNum(c.p95)}
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
