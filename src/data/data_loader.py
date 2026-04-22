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

    # Ensure required columns are present.
    target_col = config['data'].get('target_col', 'ICD_CODES')
    required_cols = ['admission_note', target_col]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Expected column '{col}' not found in data. Evaluation may fail.")

    return df
