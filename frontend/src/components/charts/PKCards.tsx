import { formatNum, confidenceColor } from "../../lib/utils";

interface CardData {
  label: string;
  value: number;
  unit: string;
  p5?: number;
  p95?: number;
}

interface Props {
  cards: CardData[];
  confidence?: string;
}

export type { CardData };

export default function PKCards({ cards, confidence }: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4"
        >
          <div className="text-xs text-[var(--text-muted)] mb-1">{c.label}</div>
          <div className="text-2xl font-semibold text-[var(--text)]">
            {formatNum(c.value)}
            <span className="text-sm font-normal text-[var(--text-muted)] ml-1">
              {c.unit}
            </span>
          </div>
          {c.p5 != null && c.p95 != null && (
            <div className="text-xs text-[var(--text-muted)] mt-1">
              CI: {formatNum(c.p5)} &ndash; {formatNum(c.p95)} {c.unit}
            </div>
          )}
          {confidence && (
            <div className="mt-2 h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: confidence === "high" ? "100%" : confidence === "medium" ? "60%" : "30%",
                  backgroundColor: confidenceColor(confidence),
                }}
              />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
