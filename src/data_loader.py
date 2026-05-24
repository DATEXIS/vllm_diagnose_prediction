import os
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith('.pq') or file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file format: {file_path}. Use .pq, .parquet, or .csv")


def _check_required_cols(df: pd.DataFrame, target_col: str):
    for col in ['admission_note', target_col]:
        if col not in df.columns:
            logger.warning(f"Expected column '{col}' not found. Evaluation may fail.")


def _proportional_sample(df: pd.DataFrame, stratum_col: str, size: int, seed: int) -> pd.DataFrame:
    """Sample `size` rows proportionally across strata, topping up any rounding shortfall."""
    groups = df.groupby(stratum_col, group_keys=False)
    per_stratum = {k: max(1, round(size * len(g) / len(df))) for k, g in groups}
    sampled = groups.apply(lambda g: g.sample(n=per_stratum[g.name], random_state=seed))
    if len(sampled) > size:
        sampled = sampled.sample(n=size, random_state=seed)
    elif len(sampled) < size:
        remaining = df.drop(index=sampled.index)
        top_up = remaining.sample(n=size - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, top_up])
    return sampled


def _create_eval_slice(source_path: str, eval_cfg: dict) -> pd.DataFrame:
    df = _read_file(source_path)

    source_split = eval_cfg.get('source_split', 'val')
    if source_split != 'all' and 'split' in df.columns:
        df = df[df['split'] == source_split].copy()
        logger.info(f"Filtered to '{source_split}' split: {len(df)} patients")

    size = eval_cfg.get('size', 25)
    seed = eval_cfg.get('seed', 42)
    stratify_by = eval_cfg.get('stratify_by', 'complexity')

    if len(df) < size:
        logger.warning(f"Source has only {len(df)} patients; returning all of them.")
        return df.reset_index(drop=True)

    rng = np.random.default_rng(seed)

    if stratify_by == 'note_length':
        df['_len'] = df['admission_note'].str.len()
        df['_stratum'] = pd.qcut(df['_len'], q=3, labels=['short', 'medium', 'long'])
        sampled = _proportional_sample(df, '_stratum', size, seed)
        drop_cols = [c for c in ['_len', '_stratum'] if c in sampled.columns]
        return sampled.drop(columns=drop_cols).reset_index(drop=True)

    elif stratify_by == 'complexity':
        tmp_cols = ['_len', '_n_codes', '_len_bin', '_code_bin', '_stratum']
        df['_len'] = df['admission_note'].str.len()
        df['_n_codes'] = df['ICD_CODES'].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
        df['_len_bin'] = pd.qcut(df['_len'], q=3, labels=['short', 'medium', 'long'])
        df['_code_bin'] = pd.qcut(df['_n_codes'], q=2, labels=['simple', 'complex'], duplicates='drop')
        df['_stratum'] = df['_len_bin'].astype(str) + '_' + df['_code_bin'].astype(str)
        sampled = _proportional_sample(df, '_stratum', size, seed)
        drop_cols = [c for c in tmp_cols if c in sampled.columns]
        return sampled.drop(columns=drop_cols).reset_index(drop=True)

    else:  # random
        return df.sample(n=size, random_state=seed).reset_index(drop=True)


def load_eval_slice(config: dict) -> pd.DataFrame:
    """Load the fixed eval slice, creating and saving it if it doesn't exist yet."""
    eval_cfg = config['data']['eval_slice']
    slice_path = eval_cfg['path']

    if os.path.exists(slice_path):
        logger.info(f"Loading existing eval slice from {slice_path} (delete file to regenerate)")
        df = _read_file(slice_path)
        logger.info(f"Eval slice: {len(df)} patients")
    else:
        source_path = config['data']['patients_file']
        logger.info(f"Eval slice not found at {slice_path} — creating from {source_path}")
        df = _create_eval_slice(source_path, eval_cfg)
        os.makedirs(os.path.dirname(slice_path), exist_ok=True)
        df.to_parquet(slice_path, index=False)
        logger.info(f"Saved eval slice ({len(df)} patients) to {slice_path}")

    target_col = config['data'].get('target_col', 'ICD_CODES')
    _check_required_cols(df, target_col)
    return df


def load_patients(config: dict) -> pd.DataFrame:
    """Load patients from the configured file, or the fixed eval slice if enabled."""
    eval_cfg = config['data'].get('eval_slice', {})
    if eval_cfg.get('enabled', False):
        return load_eval_slice(config)

    file_path = config['data']['patients_file']
    sample_size = config['data'].get('sample_size')

    logger.info(f"Loading patient data from {file_path}")
    try:
        df = _read_file(file_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    if sample_size is not None and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} patients from a total of {len(df)}")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        logger.info(f"Loaded {len(df)} patients.")

    target_col = config['data'].get('target_col', 'ICD_CODES')
    _check_required_cols(df, target_col)
    return df
