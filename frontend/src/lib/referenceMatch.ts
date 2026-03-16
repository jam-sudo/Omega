import { referenceDatabase, ReferenceData } from "../data/referenceData";

export function findReference(smiles: string): ReferenceData | null {
  for (const [, ref] of Object.entries(referenceDatabase)) {
    if (ref.smiles === smiles) return ref;
  }
  return null;
}
