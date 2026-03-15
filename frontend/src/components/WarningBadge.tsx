import { AlertTriangle, ShieldAlert, ShieldCheck } from "lucide-react";

interface Props {
  warnings: string[];
  riskLevel?: string;
}

function riskConfig(level: string) {
  switch (level) {
    case "high":
      return {
        border: "border-red-500/20",
        bg: "bg-red-500/5",
        text: "text-red-400",
        icon: ShieldAlert,
        label: "High Risk",
      };
    case "medium":
      return {
        border: "border-yellow-500/20",
        bg: "bg-yellow-500/5",
        text: "text-yellow-400",
        icon: AlertTriangle,
        label: "Medium Risk",
      };
    case "low":
      return {
        border: "border-green-500/20",
        bg: "bg-green-500/5",
        text: "text-green-400",
        icon: ShieldCheck,
        label: "Low Risk",
      };
    default:
      return {
        border: "border-[var(--border)]",
        bg: "bg-[var(--surface)]",
        text: "text-[var(--text-muted)]",
        icon: AlertTriangle,
        label: level,
      };
  }
}

export default function WarningBadge({ warnings, riskLevel }: Props) {
  if (warnings.length === 0 && !riskLevel) return null;

  const risk = riskLevel ? riskConfig(riskLevel) : null;
  const RiskIcon = risk?.icon || AlertTriangle;

  return (
    <div
      className={`rounded-lg border p-4 ${risk ? `${risk.border} ${risk.bg}` : "border-yellow-500/20 bg-yellow-500/5"}`}
    >
      {/* Risk level header */}
      {risk && (
        <div className={`flex items-center gap-2 mb-3 ${risk.text}`}>
          <RiskIcon size={16} strokeWidth={2} />
          <span className="text-sm font-medium">{risk.label}</span>
        </div>
      )}

      {/* Warning list */}
      {warnings.length > 0 && (
        <ul className="space-y-1.5">
          {warnings.map((w, i) => (
            <li
              key={i}
              className="text-sm text-[var(--text-muted)] flex items-start gap-2"
            >
              <AlertTriangle
                size={13}
                className="text-yellow-400 mt-0.5 shrink-0"
                strokeWidth={2}
              />
              <span>{w}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
