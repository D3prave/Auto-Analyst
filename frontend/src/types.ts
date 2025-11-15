// src/types.ts

export type ColumnTypeLabel =
  | "numeric"
  | "categorical"
  | "datetime"
  | "id"
  | "text"
  | "boolean";

export interface UploadResponse {
  dataset_id: string;
  n_rows: number;
  n_cols: number;
  columns: string[];
}

export interface ColumnProfile {
  name: string;
  inferred_type: string;
  effective_type: string;
  overridden_type?: ColumnTypeLabel | null;
  missing_count: number;
  missing_pct: number;
  distinct_count: number;
  numeric_summary?: {
    count: number;
    mean: number;
    std: number;
    min: number;
    q1: number;
    median: number;
    q3: number;
    max: number;
    iqr?: number;
    outlier_count?: number;
  };
  categorical_summary?: {
    top_values: string[];
    top_counts: number[];
    unique: number;
  };
  datetime_summary?: {
    min: string | null;
    max: string | null;
  };
  plot_suggested?: boolean;
}

export interface Profile {
  n_rows: number;
  n_cols: number;
  columns: { [name: string]: ColumnProfile };
  missing: {
    rows_with_missing: number;
    missing_by_column: { [name: string]: number };
  };
}

export interface PlotInfoNumeric {
  column: string;
  histogram: string | null;
  boxplot: string | null;
}

export interface PlotInfoCategorical {
  column: string;
  barplot: string | null;
  top_values: string[];
  top_counts: number[];
}

export interface PlotsResponse {
  numeric: PlotInfoNumeric[];
  categorical: PlotInfoCategorical[];
  correlation_heatmap: {
    path: string;
    columns: string[];
  } | null;
}

export interface EDAResponse {
  dataset_id: string;
  profile: Profile;
  plots: PlotsResponse;
}

export interface ModelingResponse {
  task_type: "classification" | "regression";
  target: string;

  // new fields from backend:
  best_model: string; // "logreg" | "rf" | "gb" ...
  best_model_class: string; // e.g. "RandomForestClassifier"
  best_model_params: { [key: string]: any };
  models: {
    [name: string]: {
      metrics: { [metric: string]: number };
      class_name?: string;
    };
  };

  n_rows: number;
  n_features: number;
  test_size: number;
  random_state: number;
  metrics: { [key: string]: number };
  feature_importances: { feature: string; importance: number }[] | null;
  preprocessing: {
    numeric_features: string[];
    categorical_features: string[];
  };
}

export interface InsightsResponse {
  dataset_id: string;
  target: string | null;
  insights: {
    overview: string;
    missingness: string;
    target_balance: string;
    modeling: string;
    feature_importance: string;
  };
}