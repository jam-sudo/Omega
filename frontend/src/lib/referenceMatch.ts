import type { ReferenceData } from "../data/referenceData";

let _cache: Record<string, ReferenceData> | null = null;

async function loadDatabase(): Promise<Record<string, ReferenceData>> {
  if (_cache) return _cache;
  const { referenceDatabase } = await import("../data/referenceData");
  _cache = referenceDatabase;
  return _cache;
}

export async function findReference(smiles: string): Promise<ReferenceData | null> {
  const db = await loadDatabase();
  for (const [, ref] of Object.entries(db)) {
    if (ref.smiles === smiles) return ref;
  }
  return null;
}
