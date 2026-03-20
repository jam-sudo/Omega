import type { ReferenceData } from "../data/referenceData";

let _cache: Record<string, ReferenceData> | null = null;
let _smilesIndex: Map<string, ReferenceData[]> | null = null;

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

/** Trigram Jaccard similarity on bond-normalized SMILES.
 *  Strips bond-order characters (=, #) before comparison so that
 *  kekulized (C1=CC=C) and aromatic (c1ccc) representations of the
 *  same molecule produce high similarity, while structurally
 *  different molecules with the same heavy-atom formula score low. */
function smilesJaccard(a: string, b: string): number {
  const trigrams = (s: string): Set<string> => {
    const norm = s.replace(/[=#]/g, "").toLowerCase();
    const set = new Set<string>();
    for (let i = 0; i <= norm.length - 3; i++) set.add(norm.substring(i, i + 3));
    return set;
  };
  const ta = trigrams(a);
  const tb = trigrams(b);
  let inter = 0;
  for (const t of ta) if (tb.has(t)) inter++;
  const union = ta.size + tb.size - inter;
  return union === 0 ? 0 : inter / union;
}

async function loadDatabase(): Promise<Record<string, ReferenceData>> {
  if (_cache) return _cache;
  const { referenceDatabase } = await import("../data/referenceData");
  _cache = referenceDatabase;

  // Build SMILES fingerprint index for fuzzy matching (store arrays for collisions)
  _smilesIndex = new Map();
  for (const ref of Object.values(referenceDatabase)) {
    if (ref.smiles) {
      const fp = atomFingerprint(ref.smiles);
      const existing = _smilesIndex.get(fp);
      if (existing) {
        existing.push(ref);
      } else {
        _smilesIndex.set(fp, [ref]);
      }
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
    const candidates = _smilesIndex.get(fp);
    if (candidates) {
      if (candidates.length === 1) return candidates[0];
      // Multiple candidates with same formula — pick the most similar SMILES
      let best: ReferenceData | null = null;
      let bestScore = -1;
      for (const c of candidates) {
        const score = smilesJaccard(smiles, c.smiles);
        if (score > bestScore) {
          bestScore = score;
          best = c;
        }
      }
      if (best) return best;
    }
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
