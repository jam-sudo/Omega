import { useQuery } from "@tanstack/react-query";

export default function StatusBar() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const start = performance.now();
      const res = await fetch("/api/health");
      const latency = Math.round(performance.now() - start);
      if (!res.ok) throw new Error("API unreachable");
      return { connected: true, latency };
    },
    refetchInterval: 30_000,
  });

  const connected = data?.connected && !isError;

  return (
    <footer
      className="h-8 border-t flex items-center px-4 text-xs gap-4"
      style={{ background: "var(--surface)", color: "var(--text-muted)" }}
    >
      <span>
        API:{" "}
        <span className={connected ? "text-green-400" : "text-red-400"}>
          {"\u25CF"}
        </span>{" "}
        {connected ? "connected" : "disconnected"}
      </span>
      {data?.latency && <span>Last: {data.latency}ms</span>}
      <span className="ml-auto">Omega v0.1</span>
    </footer>
  );
}
