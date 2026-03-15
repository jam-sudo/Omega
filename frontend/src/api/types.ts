export interface PredictRequest {
  smiles: string;
  dose_mg?: number;
  route?: string;
  duration_h?: number;
}

export interface FullPredictResponse {
  drug_name: string;
  smiles: string;
  cmax_mg_L: number;
  tmax_h: number;
  auc0t_mg_h_L: number;
  t_half_h: number;
  time_h: number[];
  cp_mg_L: number[];
  confidence: string;
  warnings: string[];
  adme: Record<string, number | string>;
  cmax_p5?: number;
  cmax_p50?: number;
  cmax_p95?: number;
  auc_p5?: number;
  auc_p50?: number;
  auc_p95?: number;
  risk_flags?: Record<string, boolean>;
  overall_risk_level?: string;
}

export interface DrugRequest {
  name: string;
  mw: number;
  logP: number;
  fup: number;
  rbp: number;
  clint_hepatic_L_per_h: number;
  peff: number;
  dose_mg: number;
}

export interface DDISimulateRequest {
  victim_drug: DrugRequest;
  perpetrator_name: string;
  perpetrator_ki_uM: number;
  perpetrator_target_enzyme: string;
  perpetrator_mechanism?: string;
  perpetrator_cmax_uM?: number;
}

export interface DDIResponse {
  auc_ratio: number;
  cmax_ratio: number;
  interaction_magnitude: string;
  time_h: number[];
  cp_alone: number[];
  cp_with_inhibitor: number[];
}

export interface DoseOptimizeRequest {
  drug: DrugRequest;
  cmin_mg_L: number;
  cmax_mg_L: number;
  dose_range_mg: [number, number];
}

export interface DoseOptimizeResponse {
  optimal_dose_mg: number;
  css_max: number;
  css_min: number;
  auc_ss: number;
}

export interface PopulationRequest {
  drug: DrugRequest;
  n_subjects: number;
  dose_mg: number;
  route?: string;
}

export interface PopulationResponse {
  n_subjects: number;
  cmax_median: number;
  cmax_mean: number;
  cmax_cv_pct: number;
  cmax_p5: number;
  cmax_p50: number;
  cmax_p95: number;
  auc_median: number;
  auc_mean: number;
  auc_cv_pct: number;
}

export interface HealthResponse {
  status: string;
}
