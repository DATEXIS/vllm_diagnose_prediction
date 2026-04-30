import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_patients(config: dict) -> pd.DataFrame:
    """Loads patients from the configured file and applies sampling."""
    file_path = config['data']['patients_file']
    sample_size = config['data'].get('sample_size')
    
    logger.info(f"Loading patient data from {file_path}")
    try:
        if file_path.endswith('.pq') or file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .pq, .parquet, or .csv")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    if sample_size is not None and len(df) > sample_size:
        logger.info(f"Sampling {sample_size} patients from a total of {len(df)}")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        logger.info(f"Loaded {len(df)} patients.")

    target_col = config['data'].get('target_col', 'ICD_CODES')
    admission_col = config['data'].get('admission_col', 'admission_note')
    discharge_col = config['data'].get('discharge_col', 'discharge_note')

    rename_map = {}
    if admission_col != 'admission_note' and admission_col in df.columns:
        rename_map[admission_col] = 'admission_note'
    if discharge_col and discharge_col != 'discharge_note' and discharge_col in df.columns:
        rename_map[discharge_col] = 'discharge_note'
    if target_col != 'ICD_CODES' and target_col in df.columns:
        rename_map[target_col] = 'ICD_CODES'

    if rename_map:
        logger.info(f"Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)

    return df
