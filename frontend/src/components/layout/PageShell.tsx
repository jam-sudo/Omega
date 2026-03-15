import { useEffect } from "react";

interface Props {
  title: string;
  children: React.ReactNode;
}

export default function PageShell({ title, children }: Props) {
  useEffect(() => {
    document.title = `${title} — Omega PBPK`;
  }, [title]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6">{title}</h1>
      {children}
    </div>
  );
}
