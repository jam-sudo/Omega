interface Props {
  message: string;
}

export default function ErrorAlert({ message }: Props) {
  return (
    <div className="border border-red-500/30 rounded-md px-4 py-3 text-sm text-red-400 mt-4">
      {message}
    </div>
  );
}
