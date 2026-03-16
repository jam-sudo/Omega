import type { ReferenceData } from "../data/referenceData";

let _cache: Record<string, ReferenceData> | null = null;
let _smilesIndex: Map<string, ReferenceData> | null = null;

/** Extract atom counts from SMILES for formula-based matching.
 *  Converts all atoms to uppercase (aromatic c→C, n→N etc.)
 *  so different kekulization of the same molecule matches. */
function atomFingerprint(smiles: string): string {
  // Remove everything except atom letters
  const cleaned = smiles.replace(/[@@/\\+\-\d[\]()#%.=:]/g, "");
  const counts: Record<string, number> = {};
  let i = 0;
  while (i < cleaned.length) {
    const ch = cleaned[i];
    if (ch >= "A" && ch <= "Z") {
      // Uppercase: possibly two-letter element (Cl, Br, etc.)
      if (i + 1 < cleaned.length && cleaned[i + 1] >= "a" && cleaned[i + 1] <= "z" && !"cnos".includes(cleaned[i + 1])) {
        const el = ch + cleaned[i + 1];
        counts[el] = (counts[el] || 0) + 1;
        i += 2;
      } else {
        counts[ch] = (counts[ch] || 0) + 1;
        i += 1;
      }
    } else if ("cnos".includes(ch)) {
      // Aromatic atom → count as uppercase equivalent
      const upper = ch.toUpperCase();
      counts[upper] = (counts[upper] || 0) + 1;
      i += 1;
    } else {
      i += 1;
    }
  }
  return Object.entries(counts)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([el, n]) => `${el}${n}`)
    .join("");
}

async function loadDatabase(): Promise<Record<string, ReferenceData>> {
  if (_cache) return _cache;
  const { referenceDatabase } = await import("../data/referenceData");
  _cache = referenceDatabase;

  // Build SMILES fingerprint index for fuzzy matching
  _smilesIndex = new Map();
  for (const ref of Object.values(referenceDatabase)) {
    if (ref.smiles) {
      _smilesIndex.set(atomFingerprint(ref.smiles), ref);
    }
  }

  return _cache;
}

export async function findReference(
  smiles: string,
  drugName?: string,
): Promise<ReferenceData | null> {
  const db = await loadDatabase();

  // 1. Exact SMILES match
  for (const ref of Object.values(db)) {
    if (ref.smiles === smiles) return ref;
  }

  // 2. Fingerprint match (handles different SMILES representations of same molecule)
  if (_smilesIndex) {
    const fp = atomFingerprint(smiles);
    const match = _smilesIndex.get(fp);
    if (match) return match;
  }

  // 3. Drug name fallback (if API returns a drug name)
  if (drugName) {
    const nameLower = drugName.toLowerCase().trim();
    for (const [key, ref] of Object.entries(db)) {
      if (key.toLowerCase() === nameLower || ref.name.toLowerCase() === nameLower) {
        return ref;
      }
    }
  }

  return null;
}
