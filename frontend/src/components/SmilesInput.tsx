import { useEffect, useRef } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  showPreview?: boolean;
}

export default function SmilesInput({ value, onChange, showPreview = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!showPreview || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = "#141414";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!value.trim()) return;

    let cancelled = false;

    (async () => {
      try {
        const SmilesDrawer = (await import("smiles-drawer")).default;
        if (cancelled) return;

        const drawer = new SmilesDrawer.Drawer({
          width: 200,
          height: 150,
          themes: {
            dark: {
              C: "#fafafa",
              O: "#ef4444",
              N: "#3b82f6",
              S: "#eab308",
              H: "#a3a3a3",
              BACKGROUND: "#141414",
            },
          },
        });

        SmilesDrawer.parse(value, (tree: unknown) => {
          if (cancelled) return;
          try {
            drawer.draw(tree, canvas, "dark");
          } catch {
            // Invalid SMILES — canvas stays cleared
          }
        });
      } catch {
        // smiles-drawer import failed — skip preview
      }
    })();

    return () => { cancelled = true; };
  }, [value, showPreview]);

  return (
    <div className="flex items-start gap-4">
      <div className="flex-1">
        <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
          SMILES
        </label>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
          className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm font-mono text-[var(--text)] placeholder:text-[var(--text-muted)] focus:outline-none focus:ring-1 focus:ring-blue-500"
          spellCheck={false}
          autoComplete="off"
        />
      </div>
      {showPreview && (
        <canvas
          ref={canvasRef}
          width={200}
          height={150}
          className="rounded-md border border-[var(--border)] flex-shrink-0"
          style={{ background: "#141414" }}
        />
      )}
    </div>
  );
}
