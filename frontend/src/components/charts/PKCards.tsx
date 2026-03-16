import { TrendingUp, Clock, BarChart3, Timer } from "lucide-react";
import { formatNum, confidenceColor } from "../../lib/utils";
import type { LucideIcon } from "lucide-react";

interface CardData {
  label: string;
  value: number;
  unit: string;
  p5?: number;
  p95?: number;
  reference?: number;
  foldError?: number;
}

interface Props {
  cards: CardData[];
  confidence?: string;
}

export type { CardData };

const CARD_ICONS: Record<string, LucideIcon> = {
  Cmax: TrendingUp,
  Tmax: Clock,
  "t\u00BD": Timer,
};

export default function PKCards({ cards, confidence }: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => {
        const Icon = CARD_ICONS[c.label] || BarChart3;
        return (
          <div
            key={c.label}
            className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 hover:border-blue-500/20 transition-colors group"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide">
                {c.label}
              </span>
              <Icon
                size={15}
                className="text-[var(--text-muted)] group-hover:text-blue-400 transition-colors"
                strokeWidth={1.75}
              />
            </div>
            <div className="text-2xl font-semibold text-[var(--text)] tabular-nums">
              {formatNum(c.value)}
              <span className="text-sm font-normal text-[var(--text-muted)] ml-1">
                {c.unit}
              </span>
            </div>
            {c.p5 != null && c.p95 != null && (
              <div className="text-xs text-[var(--text-muted)] mt-1.5 tabular-nums">
                90% CI: {formatNum(c.p5)} – {formatNum(c.p95)}
              </div>
            )}
            {c.reference != null && c.foldError != null && (
              <div className="flex items-center gap-1.5 mt-1.5">
                <span className="text-xs text-[var(--text-muted)] tabular-nums">
                  Ref: {formatNum(c.reference)}
                </span>
                <span
                  className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${
                    c.foldError <= 2
                      ? "bg-green-500/15 text-green-400"
                      : c.foldError <= 3
                        ? "bg-yellow-500/15 text-yellow-400"
                        : "bg-red-500/15 text-red-400"
                  }`}
                >
                  {c.foldError <= 2 ? "\u2713 " : c.foldError <= 3 ? "~ " : "\u2717 "}
                  {formatNum(c.foldError)}x
                </span>
              </div>
            )}
            {confidence && (
              <div className="mt-2.5 h-1 rounded-full bg-[var(--border)] overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width:
                      confidence === "high"
                        ? "100%"
                        : confidence === "medium"
                          ? "60%"
                          : "30%",
                    backgroundColor: confidenceColor(confidence),
                  }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
