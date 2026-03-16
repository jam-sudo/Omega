import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  showPreview?: boolean;
}

const EXAMPLES = [
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
  { name: "Acetaminophen", smiles: "CC(=O)Nc1ccc(O)cc1" },
  { name: "Metoprolol", smiles: "COCCc1ccc(OCC(O)CNC(C)C)cc1" },
  { name: "Warfarin", smiles: "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O" },
  { name: "Diazepam", smiles: "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21" },
];

export default function SmilesInput({ value, onChange, showPreview = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showExamples, setShowExamples] = useState(false);

  useEffect(() => {
    if (!showPreview || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // High-DPI: scale canvas buffer for sharp rendering on retina displays
    const CSS_W = 240;
    const CSS_H = 180;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CSS_W * dpr;
    canvas.height = CSS_H * dpr;
    canvas.style.width = `${CSS_W}px`;
    canvas.style.height = `${CSS_H}px`;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#141414";
    ctx.fillRect(0, 0, CSS_W, CSS_H);

    if (!value.trim()) return;

    let cancelled = false;

    (async () => {
      try {
        const SmilesDrawer = (await import("smiles-drawer")).default;
        if (cancelled) return;

        const drawer = new SmilesDrawer.Drawer({
          width: CSS_W,
          height: CSS_H,
          bondThickness: 1.5,
          fontSizeLarge: 11,
          fontSizeSmall: 7,
          padding: 20,
          themes: {
            dark: {
              C: "#e4e4e7",
              O: "#f87171",
              N: "#60a5fa",
              S: "#facc15",
              F: "#34d399",
              Cl: "#34d399",
              Br: "#a78bfa",
              P: "#fb923c",
              H: "#71717a",
              BACKGROUND: "#141414",
            },
          },
        });

        SmilesDrawer.parse(value, (tree: unknown) => {
          if (cancelled) return;
          try {
            drawer.draw(tree, canvas, "dark");
          } catch {
            // Invalid SMILES
          }
        });
      } catch {
        // smiles-drawer import failed
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [value, showPreview]);

  return (
    <div className="flex items-start gap-4">
      <div className="flex-1 relative">
        <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
          SMILES
        </label>
        <div className="flex gap-1">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Enter SMILES or select an example →"
            className="flex-1 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm font-mono text-[var(--text)] placeholder:text-[var(--text-muted)]/50 focus:outline-none focus:ring-1 focus:ring-blue-500"
            spellCheck={false}
            autoComplete="off"
          />
          <div className="relative">
            <button
              onClick={() => setShowExamples(!showExamples)}
              className="h-full px-2 rounded-md border border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5 transition-colors"
              title="Example drugs"
            >
              <ChevronDown size={14} />
            </button>
            {showExamples && (
              <div className="absolute right-0 top-full mt-1 w-52 rounded-lg border border-[var(--border)] bg-[#1a1a1a] shadow-xl z-50 py-1">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex.name}
                    onClick={() => {
                      onChange(ex.smiles);
                      setShowExamples(false);
                    }}
                    className="w-full text-left px-3 py-1.5 text-sm hover:bg-white/5 transition-colors flex justify-between items-center"
                  >
                    <span className="text-[var(--text)]">{ex.name}</span>
                    <span className="text-[10px] text-[var(--text-muted)] font-mono truncate max-w-[80px]">
                      {ex.smiles}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      {showPreview && (
        <div className="shrink-0">
          <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
            Structure
          </label>
          <canvas
            ref={canvasRef}
            className="rounded-lg border border-[var(--border)]"
            style={{ background: "#141414", width: 240, height: 180 }}
          />
        </div>
      )}
    </div>
  );
}
