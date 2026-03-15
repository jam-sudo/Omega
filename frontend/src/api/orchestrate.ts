import type { FullPredictResponse, DrugRequest } from "./types";

export function admeToDrugRequest(
  response: FullPredictResponse,
  dose_mg: number,
): DrugRequest {
  const adme = response.adme;
  return {
    name: response.drug_name || response.smiles.slice(0, 20),
    mw: Number(adme.mw) || 300,
    logP: Number(adme.logP) || 2,
    fup: Number(adme.fup) || 0.1,
    rbp: Number(adme.rbp) || 1.0,
    clint_hepatic_L_per_h: Number(adme.clint_hepatic_L_per_h) || 0.2,
    peff: Number(adme.peff) || 1.0,
    dose_mg,
  };
}
