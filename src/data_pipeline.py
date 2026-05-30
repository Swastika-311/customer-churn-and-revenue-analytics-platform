import joblib
import os
import pandas as pd
import numpy as np
import warnings
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# LOGGING SETUP
if os.getcwd().endswith('notebooks'):
    os.chdir('..')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("pipeline.log"),logging.StreamHandler()]
)
log=logging.getLogger(__name__)


# INGESTION
def load_data(filepath: str)->pd.DataFrame:
    log.info(f"Loading data from: {filepath}")
    df=pd.read_csv(filepath)
    log.info(f"Raw Shape: {df.shape[0]} records x {df.shape[1]} features")
    return df


# INITIAL PROFILING
def profile_data(df: pd.DataFrame)->None:
    log.info("\n DATA PROFILE")
    print(f"\nShape: {df.shape}")
    print(f"\nData Types: {df.dtypes}")
    missing = df.isnull().sum()
    print(f"\nMissing values: {missing[missing>0]}")
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"\nNumerical Summary: {df.describe()}")
    print("\nCategorical Columns Unique Counts: ")
    for col in df.select_dtypes(include="object").columns:
        print(f"\t{col}: {df[col].nunique()} unique values")


# DROP REDUNDANT COLUMNS
def drop_redundant_columns(df: pd.DataFrame)-> pd.DataFrame:
    cols_to_drop=["Count", "Country", "State", "Lat Long", "Churn Label"]
    existing=[c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    log.info(f"Dropped Redundant Coloumns: {existing}")
    return df


# RENAME COLUMNS
def rename_columns(df: pd.DataFrame)->pd.DataFrame:
    df.columns=(
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ",'_')
        .str.replace(r"[^a-z0-9_]","",regex=True)
    )
    log.info("Columns normalized to snake_case")
    return df


# FIX DATA TYPES
def fix_data_types(df: pd.DataFrame)-> pd.DataFrame:
    df['total_charges']=pd.to_numeric(df['total_charges'],errors='coerce')
    log.info(f"total_charges NaNs after coerce: {df['total_charges'].isnull().sum()}")
    df['zip_code']=df['zip_code'].astype(str).str.zfill(5)
    for col in ['tenure_months','churn_value','churn_score','cltv']:
        df[col]=df[col].astype(int)
    log.info("Data types fixed")
    return df


# HANDLE MISSING VALUES
def handle_missing_values(df: pd.DataFrame)->pd.DataFrame:
    mask = df['total_charges'].isnull()
    df.loc[mask, 'total_charges']=df.loc[mask, 'monthly_charges']
    log.info(f"Imputed {mask.sum()} total_charges values with monthly_charges")
    df['churn_reason']=df['churn_reason'].fillna("Not Churned")
    log.info('churn_reason NaNs filled with "Not Churned"')
    log.info(f"Total remaining nulls: {df.isnull().sum().sum()}")
    return df


# REMOVE DUPLICATES
def remove_duplicates(df: pd.DataFrame)->pd.DataFrame:
    n_full_dups = df.duplicated().sum()
    df = df.drop_duplicates()
    log.info(f"Removed {n_full_dups} fully duplicated rows")
    # print(f"Removed {n_full_dups} fully duplicated rows")
    n_id_dups=df.duplicated(subset=['customerid']).sum()
    if n_id_dups>0:
        log.warning(f"{n_id_dups} duplicate CustomerIDs found. Keeping first occurances")
        df =df.drop_duplicates(subset=['customerid'],keep='first')
    return df


# STANDARDIZE CATEGORIACL VALUES
def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols=df.select_dtypes(include='object').columns.tolist()
    exclude={'customerid','churn_reason','zip_code'}
    for col in cat_cols:
        if col not in exclude:
            df[col]=df[col].str.strip().str.title()
    service_cols=['online_security','online_backup',"device_protection", "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"]
    for col in service_cols:
        if col in df.columns:
            df[col]=df[col].replace({'No Internet Service' : 'No', 'No Phone Service' : 'No'})
    log.info("Categorical values strandardized.")
    # print(f"Categorical values strandardized. {df.dtypes}")
    return df


# OUTLIERS DETECTION & TREATEMENT
def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols=['monthly_charges', 'total_charges', 'tenure_months', 'churn_score', 'cltv']
    for col in numeric_cols:
        q1=df[col].quantile(0.25)
        q3=df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flag_col = f"{col}_outlier_flag"
        df[flag_col] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        n = df[flag_col].sum()
        df[col] = df[col].clip(lower = lower, upper = upper)
        if n:
            log.info(f"{col}: {n} outliers flagged and capped.")
            # print(f"{col}: {n} outliers flagged and capped.")
    return df


# FEATURE ENGINEERING
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['revenue_per_month'] = (df['total_charges'] / (df['tenure_months'] + 1)).round(2)
    df['is_long_term_customer'] = (df['tenure_months'] >= 24).astype(int)
    df['churn_risk_band'] = pd.cut(
        df['churn_score'], 
        bins=[0, 40, 60, 80, 100], 
        labels=['Low','Medium' , 'High', 'Very High'], 
        include_lowest=True)
    service_cols = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    df['num_services_subscribed'] = df[service_cols].apply(lambda col: col.str.lower() == 'yes').sum(axis=1)
    contract_map = {'Month-To-Month': 0, 'One Year': 1, 'Two Year': 2}
    df['contract_numeric'] = df['contract'].map(contract_map)
    digital_payments = {'Electronic Check', 'Bank Transfer (Automatic)', 'Credit Card (Automatic)'}
    df['is_digital_payment'] = df['payment_method'].apply(lambda x: 1 if x in digital_payments else 0)
    log.info("Feature Engineering Complete")
    return df


# SCALE + ENCODE FOR ML
def scale_and_encode(df: pd.DataFrame):
    df_ml = df.copy()
    drop_cols=["customerid", "zip_code", "latitude", "longitude",
        "churn_reason", "churn_risk_band", "contract"] + [c for c in df_ml.columns if c.endswith('_outlier_flag')]
    df_ml = df_ml.drop(columns=[c for c in drop_cols if c in df_ml.columns])

    binary_cols = ["gender", "senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]
    yn_map = {'Yes' : 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in binary_cols:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(yn_map)
    service_cols = ["multiple_lines", "online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    for col in service_cols:
        if col in df.columns:
            df_ml[col] = df[col].map({'Yes': 1, 'No': 0})

    nominal_cols = [c for c in ["internet_service", "payment_method", "city"] if c in df.columns]
    encoder = None
    if nominal_cols:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        encoder_array = encoder.fit_transform(df_ml[nominal_cols])
        encoder_cols = encoder.get_feature_names_out(nominal_cols)
        encoder_df = pd.DataFrame(encoder_array, columns=encoder_cols, index=df_ml.index)
        df_ml = df_ml.drop(columns=nominal_cols)
        df_ml = pd.concat([df_ml, encoder_df], axis=1)
        log.info(f"OneHotEncoder applied to: {nominal_cols}")

    target = df_ml['churn_value'].copy()
    feature_cols = [c for c in df_ml.columns if c != 'churn_value']

    numeric_to_scale = [
        c for c in [
            "tenure_months", "monthly_charges", 
            "total_charges", "churn_score", "cltv", 
            "revenue_per_month", "contract_numeric"]
        if c in feature_cols
    ]

    scaler = StandardScaler()
    if numeric_to_scale:
        df_ml[numeric_to_scale] = scaler.fit_transform(df_ml[numeric_to_scale])
        log.info(f"StandardScaler applied to: {numeric_to_scale}")

    return df_ml, scaler, encoder


# VALIDATION CHECKS
def validation_checks(df_clean: pd.DataFrame, df_ml: pd.DataFrame) -> None:
    assert df_clean['customerid'].nunique() == len(df_clean), "Duplicate CustomerIDs remain!"
    assert df_clean['total_charges'].isnull().sum() == 0, "NaN in total_charges!"
    assert df_clean['monthly_charges'].isnull().sum() == 0, "NaN found in monthly_charges!"
    assert df_ml.isnull().sum().sum() == 0, "NaN values found in ML dataset!"
    assert df_ml.select_dtypes(include='object').empty, "Non-numeric columns still present in Ml dataset!"
    assert 'churn_value' in df_ml.columns, "Target column (churn_value) missing from ML dataset!"
    log.info("All Validation checks passed!")


# EXPORT OUTPUTS
def export_outputs(df_clean: pd.DataFrame, df_ml: pd.DataFrame, scaler, encoder) -> None:
    if os.getcwd().endswith('notebooks'):
        os.chdir('..')
    processed_data = os.path.join('data', 'processed')
    analyst_path = os.path.join(processed_data, 'churn_cleaned_for_analyst.csv')
    ml_path = os.path.join(processed_data, 'churn_ml_ready.csv')
    df_clean.to_csv(analyst_path, index=False)
    df_ml.to_csv(ml_path, index=False)
    joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
    if encoder is not None:
        joblib.dump(encoder, os.path.join('models', 'encoder.pkl'))
    log.info(f"Analyst dataset: {analyst_path}   | Shape: {df_clean.shape}")
    log.info(f"Analyst dataset: {ml_path}   | Shape: {df_ml.shape}")


# MAIN PIPELINE 
def run_pipeline(filepath: str):
    log.info("\n=== PIPELINE START ===\n")
    df = load_data(filepath)
    profile_data(df)
    df = drop_redundant_columns(df)
    df = rename_columns(df)
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = standardize_categorical_values(df)
    df = treat_outliers(df)
    df = engineer_features(df)

    df_clean = df.copy()
    df_ml, scaler, encoder = scale_and_encode(df)

    validation_checks(df_clean, df_ml)
    if os.getcwd().endswith('notebooks'):
        os.chdir('..')
    export_outputs(df_clean, df_ml, scaler, encoder)

    log.info("\n=== PIPELINE COMPLETE ===\n")
    return df_clean, df_ml, scaler, encoder


# ENTRY POINT
if __name__=='__main__':
    if os.getcwd().endswith('notebooks'):
        os.chdir('..')
    filepath = os.path.join('data', 'raw', 'data.csv')
    df_clean, df_ml, scaler, encoder = run_pipeline(filepath)
    print("\n===== ANALYST DATASET PREVIEW =====")
    print(df_clean.head(3).to_string())
    print(f"\nShape: {df_clean.shape}")
    print("\n===== ML-READY DATASET PREVIEW (first 15 cols) =====")
    print(df_ml.head(3).iloc[:, :15].to_string())
    print(f"\nFull Shape: {df_ml.shape}")
    print(f"\nTarget distribution:\n{df_ml['churn_value'].value_counts()}")






