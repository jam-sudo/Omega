interface Props {
  message: string;
}

export default function ErrorAlert({ message }: Props) {
  return (
    <div className="border border-red-500/30 bg-red-500/10 rounded-md px-4 py-3 text-sm text-red-400 flex items-center gap-2 mt-4">
      <span>⚠</span>
      <span>{message}</span>
    </div>
  );
}
