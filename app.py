import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, davies_bouldin_score)
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, RFE
from imblearn.over_sampling import SMOTE
import joblib
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STATE MANAGEMENT CLASS - Better organization of global state
# ============================================================================

class AppState:
    """Centralized state management for the application"""
    def __init__(self):
        self.current_df = None
        self.df_history = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_model = None
        self.model_type = None
        self.feature_importance_data = None
        self.max_history = 10
    
    def set_dataframe(self, df):
        """Set the current dataframe"""
        self.current_df = df.copy() if df is not None else None
    
    def save_to_history(self):
        """Save current state to history"""
        if self.current_df is not None:
            self.df_history.append(self.current_df.copy())
            if len(self.df_history) > self.max_history:
                self.df_history.pop(0)
    
    def undo(self):
        """Undo last action"""
        if len(self.df_history) > 1:
            self.df_history.pop()
            self.current_df = self.df_history[-1].copy()
            return True, "✓ Undo successful"
        return False, "Cannot undo. No previous state available."
    
    def reset(self):
        """Reset all state"""
        self.current_df = None
        self.df_history = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_model = None
        self.model_type = None
        self.feature_importance_data = None
    
    def get_column_lists(self):
        """Get column lists for dropdowns"""
        if self.current_df is None:
            return [], [], []
        
        all_cols = self.current_df.columns.tolist()
        numeric_cols = self.current_df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = self.current_df.select_dtypes(include=['object']).columns.tolist()
        
        return all_cols, numeric_cols, text_cols

# Initialize global state
app_state = AppState()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def cleanup_plots():
    """Clean up matplotlib figures to prevent memory leaks"""
    plt.close('all')
    gc.collect()

def validate_dataframe(min_rows=1, min_cols=1):
    """Validate that dataframe exists and meets minimum requirements"""
    if app_state.current_df is None:
        return False, "No dataset loaded"
    
    if len(app_state.current_df) < min_rows:
        return False, f"Dataset needs at least {min_rows} rows"
    
    if len(app_state.current_df.columns) < min_cols:
        return False, f"Dataset needs at least {min_cols} columns"
    
    return True, "Valid"

def safe_operation(func):
    """Decorator for safe operations with automatic history saving"""
    def wrapper(*args, **kwargs):
        try:
            app_state.save_to_history()
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            # Rollback on error
            if len(app_state.df_history) > 0:
                app_state.df_history.pop()
            return f"Error: {str(e)}", None
    return wrapper

# ============================================================================
# FILE LOADING FUNCTIONS - Improved with better validation
# ============================================================================

def load_file(file):
    """Load file with improved error handling and validation"""
    if file is None:
        return "No file uploaded", None, *[gr.update(choices=[]) for _ in range(14)]
    
    filename = file.name.lower()
    
    try:
        # File size check (max 500MB)
        import os
        file_size = os.path.getsize(file.name)
        max_size = 500 * 1024 * 1024  # 500MB
        
        if file_size > max_size:
            return f"File too large ({file_size/(1024**2):.1f}MB). Maximum size is 500MB", None, *[gr.update(choices=[]) for _ in range(14)]
        
        # Load based on file type
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name)
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(file.name)
        elif filename.endswith(".json"):
            df = pd.read_json(file.name)
        else:
            return "Unsupported file format. Please upload CSV, Excel, or JSON", None, *[gr.update(choices=[]) for _ in range(14)]
        
        # Validate loaded data
        if df.empty:
            return "Error: Uploaded file is empty", None, *[gr.update(choices=[]) for _ in range(14)]
        
        if len(df.columns) == 0:
            return "Error: No columns found in file", None, *[gr.update(choices=[]) for _ in range(14)]
        
        # Set dataframe and initialize history
        app_state.set_dataframe(df)
        app_state.df_history = [df.copy()]
        
        # Prepare info message
        info = f"✓ Successfully loaded | Rows: {df.shape[0]:,} | Columns: {df.shape[1]} | Size: {file_size/(1024**2):.2f}MB"
        
        # Get column lists
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Return with all dropdown updates
        return (
            info, 
            df.head(20),
            gr.update(choices=all_cols),      # missing_cols
            gr.update(choices=all_cols),      # dup_cols
            gr.update(choices=numeric_cols),  # outlier_cols
            gr.update(choices=all_cols),      # dtype_col
            gr.update(choices=text_cols),     # text_cols
            gr.update(choices=numeric_cols),  # scale_cols
            gr.update(choices=all_cols),      # uni_col
            gr.update(choices=all_cols),      # bi_col1
            gr.update(choices=all_cols),      # bi_col2
            gr.update(choices=numeric_cols),  # outlier_col_eda
            gr.update(choices=numeric_cols),  # dist_col
            gr.update(choices=all_cols),      # cat_col
        )
        
    except Exception as e:
        error_msg = f"Error loading file: {str(e)}"
        return error_msg, None, *[gr.update(choices=[]) for _ in range(14)]

def load_file_extended(file):
    """Extended file loading that updates ALL dropdowns including Feature Engineering and Model Building"""
    result = load_file(file)
    
    if result[0] and "Error" not in result[0] and "No file uploaded" not in result[0]:
        all_cols, numeric_cols, text_cols = app_state.get_column_lists()
        
        # Add additional dropdown updates for Feature Engineering and Model Building
        return result + (
            gr.update(choices=all_cols),      # fc_col1
            gr.update(choices=all_cols),      # fc_col2
            gr.update(choices=numeric_cols),  # ft_cols
            gr.update(choices=text_cols),     # enc_cols
            gr.update(choices=numeric_cols),  # bin_col
            gr.update(choices=numeric_cols),  # poly_cols
            gr.update(choices=all_cols),      # mb_target_col
        )
    else:
        return result + tuple([gr.update(choices=[]) for _ in range(7)])

def undo_last_action():
    """Undo the last data modification"""
    success, message = app_state.undo()
    
    if success:
        preview = app_state.current_df.head(20) if app_state.current_df is not None else None
        return f"{message} (History: {len(app_state.df_history)} states)", preview
    else:
        preview = app_state.current_df.head(20) if app_state.current_df is not None else None
        return message, preview

def reset_data():
    """Reset all data and state"""
    app_state.reset()
    cleanup_plots()
    return "Dataset reset. Please upload a new file.", None

def download_cleaned_data():
    """Export cleaned data to CSV"""
    if app_state.current_df is None:
        return None
    
    try:
        output_path = "cleaned_data.csv"
        app_state.current_df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None
# ============================================================================
# DATA CLEANING SUMMARY AND QUALITY REPORT
# ============================================================================

def get_cleaning_summary():
    """Generate comprehensive data quality summary"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg
    
    df = app_state.current_df
    summary = []
    
    # Basic Info
    summary.append(f"# 📊 Data Quality Report\n")
    summary.append(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    summary.append(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n")
    
    # Data Quality Score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    summary.append(f"## Quality Metrics")
    summary.append(f"**Completeness:** {completeness:.1f}%")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        summary.append(f"\n## ⚠️ Missing Values ({missing.sum():,} total)")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            summary.append(f"- **{col}**: {count:,} ({pct:.1f}%) `{bar}`")
    else:
        summary.append(f"\n## ✓ Missing Values: None")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        summary.append(f"\n## ⚠️ Duplicate Rows: {dup_count:,} ({dup_pct:.1f}%)")
    else:
        summary.append(f"\n## ✓ Duplicate Rows: None")
    
    # Data types
    summary.append(f"\n## Data Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        summary.append(f"- **{dtype}**: {count} columns")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary.append(f"\n## Numeric Columns ({len(numeric_cols)})")
        for col in numeric_cols[:5]:  # Show first 5
            summary.append(f"- **{col}**: range [{df[col].min():.2f}, {df[col].max():.2f}]")
        if len(numeric_cols) > 5:
            summary.append(f"- ... and {len(numeric_cols) - 5} more")
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        summary.append(f"\n## Categorical Columns ({len(cat_cols)})")
        for col in cat_cols[:5]:
            unique = df[col].nunique()
            summary.append(f"- **{col}**: {unique} unique values")
        if len(cat_cols) > 5:
            summary.append(f"- ... and {len(cat_cols) - 5} more")
    
    # Recommendations
    summary.append(f"\n## 💡 Recommendations")
    if missing.sum() > 0:
        summary.append(f"- 🔧 Handle missing values in {len(missing[missing > 0])} columns")
    if dup_count > 0:
        summary.append(f"- 🔧 Remove {dup_count:,} duplicate rows")
    if len(numeric_cols) > 0:
        summary.append(f"- 📊 Check numeric columns for outliers")
    
    return "\n".join(summary)

# ============================================================================
# DATA CLEANING FUNCTIONS - Improved with better error handling
# ============================================================================

def handle_missing_values(strategy, columns, fill_value):
    """Handle missing values with various strategies"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        processed_cols = []
        skipped_cols = []
        
        for col in columns:
            if col not in df.columns:
                skipped_cols.append(f"{col} (not found)")
                continue
                
            if df[col].isnull().sum() == 0:
                skipped_cols.append(f"{col} (no missing values)")
                continue
            
            if strategy == "Drop rows":
                initial_rows = len(df)
                df = df.dropna(subset=[col])
                removed = initial_rows - len(df)
                processed_cols.append(f"{col} ({removed} rows removed)")
                
            elif strategy == "Fill with mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                    processed_cols.append(col)
                else:
                    skipped_cols.append(f"{col} (not numeric)")
                    
            elif strategy == "Fill with median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                    processed_cols.append(col)
                else:
                    skipped_cols.append(f"{col} (not numeric)")
                    
            elif strategy == "Fill with mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                    processed_cols.append(col)
                else:
                    skipped_cols.append(f"{col} (no mode found)")
                    
            elif strategy == "Fill with custom value":
                try:
                    # Try to convert fill_value to appropriate type
                    if pd.api.types.is_numeric_dtype(df[col]):
                        fill_val = float(fill_value) if '.' in str(fill_value) else int(fill_value)
                    else:
                        fill_val = fill_value
                    df[col] = df[col].fillna(fill_val)
                    processed_cols.append(col)
                except ValueError:
                    skipped_cols.append(f"{col} (invalid fill value)")
                    
            elif strategy == "Forward fill":
                df[col] = df[col].ffill()
                processed_cols.append(col)
                
            elif strategy == "Backward fill":
                df[col] = df[col].bfill()
                processed_cols.append(col)
        
        app_state.current_df = df
        
        # Build status message
        status = f"✓ Missing values handled using '{strategy}'\n"
        if processed_cols:
            status += f"✓ Processed: {', '.join(processed_cols[:3])}"
            if len(processed_cols) > 3:
                status += f" and {len(processed_cols)-3} more"
        if skipped_cols:
            status += f"\n⚠️ Skipped: {', '.join(skipped_cols[:3])}"
            if len(skipped_cols) > 3:
                status += f" and {len(skipped_cols)-3} more"
        
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def remove_duplicates(subset_cols):
    """Remove duplicate rows"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    initial_count = len(df)
    
    try:
        # Validate subset columns
        if subset_cols:
            invalid_cols = [col for col in subset_cols if col not in df.columns]
            if invalid_cols:
                app_state.df_history.pop()
                return f"⚠️ Invalid columns: {', '.join(invalid_cols)}", None
            
            df = df.drop_duplicates(subset=subset_cols, keep='first')
        else:
            df = df.drop_duplicates(keep='first')
        
        removed = initial_count - len(df)
        app_state.current_df = df
        
        if removed == 0:
            return "✓ No duplicate rows found", df.head(20)
        else:
            pct = (removed / initial_count) * 100
            return f"✓ Removed {removed:,} duplicate rows ({pct:.1f}%)", df.head(20)
            
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def handle_outliers(method, columns, threshold):
    """Handle outliers using various methods"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        removed_count = 0
        processed_cols = []
        skipped_cols = []
        
        for col in columns:
            if col not in df.columns:
                skipped_cols.append(f"{col} (not found)")
                continue
                
            if not pd.api.types.is_numeric_dtype(df[col]):
                skipped_cols.append(f"{col} (not numeric)")
                continue
            
            initial = len(df)
            
            if method == "IQR Method":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
                removed = initial - len(df)
                removed_count += removed
                processed_cols.append(f"{col} ({removed} outliers)")
                
            elif method == "Z-Score Method":
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                if len(z_scores) > 0:
                    mask = np.abs(stats.zscore(df[col].fillna(df[col].median()))) < threshold
                    df = df[mask]
                    removed = initial - len(df)
                    removed_count += removed
                    processed_cols.append(f"{col} ({removed} outliers)")
                    
            elif method == "Cap outliers":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower, upper)
                processed_cols.append(f"{col} (capped)")
        
        app_state.current_df = df
        
        status = f"✓ Outliers handled using '{method}'\n"
        if processed_cols:
            status += f"✓ Processed: {', '.join(processed_cols[:3])}"
            if len(processed_cols) > 3:
                status += f" and {len(processed_cols)-3} more"
        if method != "Cap outliers" and removed_count > 0:
            status += f"\n📊 Total rows removed: {removed_count:,}"
        if skipped_cols:
            status += f"\n⚠️ Skipped: {', '.join(skipped_cols)}"
            
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

  # ============================================================================
# DATA CLEANING FUNCTIONS - Part 2
# ============================================================================

def correct_data_types(column, new_type):
    """Correct data types with better error handling"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not column:
        return "⚠️ Please select a column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        if column not in df.columns:
            app_state.df_history.pop()
            return f"⚠️ Column '{column}' not found", None
        
        original_type = df[column].dtype
        
        if new_type == "int":
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif new_type == "float":
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == "string":
            df[column] = df[column].astype(str)
        elif new_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == "category":
            df[column] = df[column].astype('category')
        elif new_type == "bool":
            df[column] = df[column].astype(bool)
        
        app_state.current_df = df
        
        # Check for coerced values
        null_count = df[column].isnull().sum()
        warning = ""
        if null_count > 0:
            warning = f"\n⚠️ Warning: {null_count} values could not be converted (set to NaN)"
        
        return f"✓ Column '{column}' converted: {original_type} → {new_type}{warning}", df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def standardize_text(columns, operation):
    """Standardize text columns"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        processed_cols = []
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if operation == "Lowercase":
                df[col] = df[col].astype(str).str.lower()
            elif operation == "Uppercase":
                df[col] = df[col].astype(str).str.upper()
            elif operation == "Title Case":
                df[col] = df[col].astype(str).str.title()
            elif operation == "Strip whitespace":
                df[col] = df[col].astype(str).str.strip()
            elif operation == "Remove special characters":
                df[col] = df[col].astype(str).str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            
            processed_cols.append(col)
        
        app_state.current_df = df
        
        status = f"✓ Text standardized: {operation}\n"
        status += f"✓ Processed {len(processed_cols)} column(s): {', '.join(processed_cols[:3])}"
        if len(processed_cols) > 3:
            status += f" and {len(processed_cols)-3} more"
        
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def scale_normalize(columns, method):
    """Scale and normalize numeric columns"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        # Validate all columns are numeric
        invalid_cols = []
        valid_cols = []
        
        for col in columns:
            if col not in df.columns:
                invalid_cols.append(f"{col} (not found)")
            elif not pd.api.types.is_numeric_dtype(df[col]):
                invalid_cols.append(f"{col} (not numeric)")
            else:
                valid_cols.append(col)
        
        if not valid_cols:
            app_state.df_history.pop()
            return "⚠️ No valid numeric columns selected", None
        
        # Apply scaling
        if method == "Standard Scaler (Z-score)":
            scaler = StandardScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
        elif method == "Min-Max Scaler (0-1)":
            scaler = MinMaxScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
        elif method == "Robust Scaler":
            scaler = RobustScaler()
            df[valid_cols] = scaler.fit_transform(df[valid_cols])
        
        app_state.current_df = df
        
        status = f"✓ Scaling applied: {method}\n"
        status += f"✓ Scaled {len(valid_cols)} column(s): {', '.join(valid_cols[:3])}"
        if len(valid_cols) > 3:
            status += f" and {len(valid_cols)-3} more"
        if invalid_cols:
            status += f"\n⚠️ Skipped: {', '.join(invalid_cols[:2])}"
        
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

  # ============================================================================
# EDA FUNCTIONS - Part 1
# ============================================================================

def understand_data():
    """Generate comprehensive data overview"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    df = app_state.current_df
    info = []
    
    info.append(f"# 📊 Dataset Overview\n")
    info.append(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    info.append(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n")
    
    # Data types summary
    info.append(f"## Data Types Summary")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        info.append(f"- **{dtype}**: {count} columns")
    
    info.append(f"\n## Column Details\n")
    
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique = df[col].nunique()
        
        info.append(f"### {col}")
        info.append(f"- **Type:** {dtype}")
        info.append(f"- **Non-Null:** {non_null:,} | **Null:** {null_count:,} ({null_pct:.1f}%)")
        info.append(f"- **Unique Values:** {unique:,}")
        
        if pd.api.types.is_numeric_dtype(df[col]):
            info.append(f"- **Range:** [{df[col].min():.2f}, {df[col].max():.2f}]")
        elif unique <= 10:
            top_vals = df[col].value_counts().head(3)
            info.append(f"- **Top Values:** {', '.join([f'{k} ({v})' for k, v in top_vals.items()])}")
        
        info.append("")
    
    return "\n".join(info), df.head(10)

def descriptive_stats(stat_type):
    """Generate descriptive statistics"""
    valid, msg = validate_dataframe()
    if not valid:
        return pd.DataFrame({"Error": [msg]})
    
    try:
        df = app_state.current_df
        
        if stat_type == "Numeric Only":
            return df.describe()
        elif stat_type == "All Columns":
            return df.describe(include='all')
        else:  # Categorical Only
            cat_df = df.select_dtypes(include=['object', 'category'])
            if cat_df.empty:
                return pd.DataFrame({"Message": ["No categorical columns found"]})
            return cat_df.describe()
            
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

def univariate_analysis(column, plot_type):
    """Perform univariate analysis with various visualizations"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    if not column:
        return None, "⚠️ Please select a column"
    
    df = app_state.current_df
    
    if column not in df.columns:
        return None, f"⚠️ Column '{column}' not found"
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df[column]):
                data = df[column].dropna()
                ax.hist(data, bins=min(30, len(data.unique())), edgecolor='black', alpha=0.7, color='steelblue')
                ax.set_xlabel(column, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(f"Histogram of {column}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add mean and median lines
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                ax.legend()
            else:
                cleanup_plots()
                return None, "⚠️ Histogram only works with numeric columns"
        
        elif plot_type == "Box Plot":
            if pd.api.types.is_numeric_dtype(df[column]):
                data = df[column].dropna()
                bp = ax.boxplot(data, vert=True, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_edgecolor('black')
                ax.set_ylabel(column, fontsize=12)
                ax.set_title(f"Box Plot of {column}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            else:
                cleanup_plots()
                return None, "⚠️ Box plot only works with numeric columns"
        
        elif plot_type == "Value Counts":
            value_counts = df[column].value_counts().head(20)
            colors = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=colors, edgecolor='black')
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"Value Counts of {column} (Top 20)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        elif plot_type == "Statistics":
            stats_text = f"# Statistics for {column}\n\n"
            
            if pd.api.types.is_numeric_dtype(df[column]):
                data = df[column].dropna()
                stats_text += f"## Descriptive Statistics\n\n"
                stats_text += f"| Metric | Value |\n|--------|-------|\n"
                stats_text += f"| **Count** | {len(data):,} |\n"
                stats_text += f"| **Mean** | {data.mean():.4f} |\n"
                stats_text += f"| **Median** | {data.median():.4f} |\n"
                stats_text += f"| **Std Dev** | {data.std():.4f} |\n"
                stats_text += f"| **Min** | {data.min():.4f} |\n"
                stats_text += f"| **25%** | {data.quantile(0.25):.4f} |\n"
                stats_text += f"| **50%** | {data.quantile(0.50):.4f} |\n"
                stats_text += f"| **75%** | {data.quantile(0.75):.4f} |\n"
                stats_text += f"| **Max** | {data.max():.4f} |\n\n"
                
                # Skewness and Kurtosis
                from scipy.stats import skew, kurtosis
                stats_text += f"## Distribution Shape\n\n"
                stats_text += f"- **Skewness:** {skew(data):.4f} "
                if abs(skew(data)) < 0.5:
                    stats_text += "(Fairly symmetric)\n"
                elif skew(data) > 0:
                    stats_text += "(Right-skewed)\n"
                else:
                    stats_text += "(Left-skewed)\n"
                stats_text += f"- **Kurtosis:** {kurtosis(data):.4f}\n\n"
            
            stats_text += f"## General Info\n\n"
            stats_text += f"- **Unique Values:** {df[column].nunique():,}\n"
            stats_text += f"- **Missing Values:** {df[column].isnull().sum():,} ({df[column].isnull().sum()/len(df)*100:.1f}%)\n"
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                top_5 = df[column].value_counts().head(5)
                stats_text += f"\n## Top 5 Values\n\n"
                for val, count in top_5.items():
                    pct = (count / len(df)) * 100
                    stats_text += f"- **{val}**: {count:,} ({pct:.1f}%)\n"
            
            cleanup_plots()
            return None, stats_text
        
        plt.tight_layout()
        return fig, ""
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

  # ============================================================================
# EDA FUNCTIONS - Part 2 (Bivariate and Multivariate Analysis)
# ============================================================================

def bivariate_analysis(col1, col2, plot_type):
    """Perform bivariate analysis"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    if not col1 or not col2:
        return None, "⚠️ Please select both columns"
    
    df = app_state.current_df
    
    if col1 not in df.columns or col2 not in df.columns:
        return None, "⚠️ One or both columns not found"
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter Plot":
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                ax.scatter(df[col1], df[col2], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                ax.set_xlabel(col1, fontsize=12)
                ax.set_ylabel(col2, fontsize=12)
                ax.set_title(f"Scatter Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add trend line
                z = np.polyfit(df[col1].dropna(), df[col2].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(df[col1], p(df[col1]), "r--", alpha=0.8, linewidth=2, label='Trend line')
                ax.legend()
            else:
                cleanup_plots()
                return None, "⚠️ Scatter plot requires numeric columns"
        
        elif plot_type == "Line Plot":
            if pd.api.types.is_numeric_dtype(df[col2]):
                # Handle if col1 is datetime
                if pd.api.types.is_datetime64_any_dtype(df[col1]):
                    sorted_df = df.sort_values(col1)
                    ax.plot(sorted_df[col1], sorted_df[col2], marker='o', markersize=4, linewidth=2)
                else:
                    ax.plot(df[col1], df[col2], marker='o', markersize=4, linewidth=2)
                ax.set_xlabel(col1, fontsize=12)
                ax.set_ylabel(col2, fontsize=12)
                ax.set_title(f"Line Plot: {col1} vs {col2}", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45, ha='right')
            else:
                cleanup_plots()
                return None, "⚠️ Line plot requires numeric Y-axis"
        
        elif plot_type == "Bar Plot":
            grouped = df.groupby(col1)[col2].mean().head(20)
            colors = plt.cm.viridis(np.linspace(0, 1, len(grouped)))
            bars = ax.bar(range(len(grouped)), grouped.values, color=colors, edgecolor='black')
            ax.set_xticks(range(len(grouped)))
            ax.set_xticklabels(grouped.index, rotation=45, ha='right')
            ax.set_xlabel(col1, fontsize=12)
            ax.set_ylabel(f"Mean of {col2}", fontsize=12)
            ax.set_title(f"Bar Plot: {col1} vs {col2} (Top 20)", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        elif plot_type == "Correlation":
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                # Calculate correlations
                pearson_corr = df[[col1, col2]].corr().iloc[0, 1]
                spearman_corr = df[[col1, col2]].corr(method='spearman').iloc[0, 1]
                
                stats_text = f"# Correlation Analysis\n\n"
                stats_text += f"## {col1} vs {col2}\n\n"
                stats_text += f"| Correlation Type | Value | Interpretation |\n"
                stats_text += f"|------------------|-------|----------------|\n"
                stats_text += f"| **Pearson** | {pearson_corr:.4f} | "
                
                if abs(pearson_corr) > 0.7:
                    stats_text += "Strong correlation |\n"
                elif abs(pearson_corr) > 0.4:
                    stats_text += "Moderate correlation |\n"
                else:
                    stats_text += "Weak correlation |\n"
                
                stats_text += f"| **Spearman** | {spearman_corr:.4f} | "
                
                if abs(spearman_corr) > 0.7:
                    stats_text += "Strong correlation |\n"
                elif abs(spearman_corr) > 0.4:
                    stats_text += "Moderate correlation |\n"
                else:
                    stats_text += "Weak correlation |\n"
                
                stats_text += f"\n## Interpretation Guide\n\n"
                stats_text += f"- **|r| > 0.7**: Strong relationship\n"
                stats_text += f"- **0.4 < |r| ≤ 0.7**: Moderate relationship\n"
                stats_text += f"- **|r| ≤ 0.4**: Weak relationship\n"
                stats_text += f"\n*Positive values indicate positive correlation, negative values indicate negative correlation*\n"
                
                cleanup_plots()
                return None, stats_text
            else:
                cleanup_plots()
                return None, "⚠️ Correlation requires numeric columns"
        
        plt.tight_layout()
        return fig, ""
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

def correlation_matrix(method):
    """Generate correlation matrix heatmap"""
    valid, msg = validate_dataframe()
    if not valid:
        cleanup_plots()
        return None
    
    try:
        numeric_df = app_state.current_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        if len(numeric_df.columns) < 2:
            return None
        
        corr = numeric_df.corr(method=method)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(12, len(corr.columns)), max(10, len(corr.columns) * 0.8)))
        
        # Create heatmap
        im = ax.imshow(corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)
        
        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                             fontsize=8)
        
        ax.set_title(f"Correlation Matrix ({method.capitalize()})", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        cleanup_plots()
        return None

def missing_value_analysis():
    """Analyze missing values with visualization"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    try:
        df = app_state.current_df
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            return None, "# ✓ No Missing Values Found!\n\nYour dataset is complete."
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, max(6, len(missing) * 0.3)))
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(missing)))
        bars = ax.barh(range(len(missing)), missing.values, color=colors, edgecolor='black')
        ax.set_yticks(range(len(missing)))
        ax.set_yticklabels(missing.index)
        ax.set_xlabel("Missing Count", fontsize=12)
        ax.set_title("Missing Values by Column", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pct = (width / len(df)) * 100
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{int(width):,} ({pct:.1f}%)',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Generate details
        details = f"# Missing Value Analysis\n\n"
        details += f"**Total Missing Cells:** {missing.sum():,}\n\n"
        details += f"**Affected Columns:** {len(missing)}\n\n"
        details += f"## Details by Column\n\n"
        
        for col, count in missing.items():
            pct = (count / len(df)) * 100
            severity = "🔴 Critical" if pct > 50 else "🟡 Moderate" if pct > 20 else "🟢 Minor"
            details += f"### {col}\n"
            details += f"- **Missing:** {count:,} ({pct:.1f}%)\n"
            details += f"- **Severity:** {severity}\n\n"
        
        return fig, details
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

  # ============================================================================
# EDA FUNCTIONS - Part 3 (Outliers, Distribution, Categorical)
# ============================================================================

def outlier_detection(column, method):
    """Detect outliers using various methods"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    if not column:
        return None, "⚠️ Please select a column"
    
    df = app_state.current_df
    
    if column not in df.columns:
        return None, f"⚠️ Column '{column}' not found"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, "⚠️ Please select a numeric column"
    
    try:
        data = df[column].dropna()
        
        if len(data) == 0:
            return None, "⚠️ No data available after removing nulls"
        
        if method == "Box Plot":
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_edgecolor('black')
            bp['medians'][0].set_color('red')
            bp['medians'][0].set_linewidth(2)
            
            # Mark outliers
            for flier in bp['fliers']:
                flier.set(marker='o', color='red', alpha=0.5, markersize=8)
            
            ax.set_ylabel(column, fontsize=12)
            ax.set_title(f"Box Plot - Outlier Detection for {column}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            plt.tight_layout()
            return fig, ""
        
        elif method == "IQR Analysis":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]
            
            stats_text = f"# IQR Outlier Analysis\n\n"
            stats_text += f"## Column: {column}\n\n"
            stats_text += f"| Statistic | Value |\n|-----------|-------|\n"
            stats_text += f"| **Q1 (25%)** | {Q1:.4f} |\n"
            stats_text += f"| **Q2 (50%)** | {data.median():.4f} |\n"
            stats_text += f"| **Q3 (75%)** | {Q3:.4f} |\n"
            stats_text += f"| **IQR** | {IQR:.4f} |\n"
            stats_text += f"| **Lower Bound** | {lower:.4f} |\n"
            stats_text += f"| **Upper Bound** | {upper:.4f} |\n\n"
            
            outlier_pct = (len(outliers) / len(data)) * 100
            severity = "🔴 High" if outlier_pct > 10 else "🟡 Moderate" if outlier_pct > 5 else "🟢 Low"
            
            stats_text += f"## Outlier Summary\n\n"
            stats_text += f"- **Total Outliers:** {len(outliers):,} / {len(data):,} ({outlier_pct:.2f}%)\n"
            stats_text += f"- **Severity:** {severity}\n"
            stats_text += f"- **Below Lower Bound:** {len(data[data < lower]):,}\n"
            stats_text += f"- **Above Upper Bound:** {len(data[data > upper]):,}\n\n"
            
            if len(outliers) > 0:
                stats_text += f"## Sample Outliers\n\n"
                sample_outliers = outliers.head(10)
                for val in sample_outliers:
                    stats_text += f"- {val:.4f}\n"
                if len(outliers) > 10:
                    stats_text += f"- ... and {len(outliers) - 10} more\n"
            
            return None, stats_text
        
        elif method == "Z-Score Analysis":
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
            stats_text = f"# Z-Score Outlier Analysis\n\n"
            stats_text += f"## Column: {column}\n\n"
            stats_text += f"| Statistic | Value |\n|-----------|-------|\n"
            stats_text += f"| **Mean** | {data.mean():.4f} |\n"
            stats_text += f"| **Std Dev** | {data.std():.4f} |\n"
            stats_text += f"| **Min Z-Score** | {z_scores.min():.4f} |\n"
            stats_text += f"| **Max Z-Score** | {z_scores.max():.4f} |\n\n"
            
            outlier_pct = (len(outliers) / len(data)) * 100
            severity = "🔴 High" if outlier_pct > 10 else "🟡 Moderate" if outlier_pct > 5 else "🟢 Low"
            
            stats_text += f"## Outlier Summary (|Z| > 3)\n\n"
            stats_text += f"- **Total Outliers:** {len(outliers):,} / {len(data):,} ({outlier_pct:.2f}%)\n"
            stats_text += f"- **Severity:** {severity}\n\n"
            
            stats_text += f"## Threshold Breakdown\n\n"
            stats_text += f"- **|Z| > 3:** {len(data[z_scores > 3]):,} ({len(data[z_scores > 3])/len(data)*100:.2f}%)\n"
            stats_text += f"- **|Z| > 2:** {len(data[z_scores > 2]):,} ({len(data[z_scores > 2])/len(data)*100:.2f}%)\n"
            stats_text += f"- **|Z| > 1:** {len(data[z_scores > 1]):,} ({len(data[z_scores > 1])/len(data)*100:.2f}%)\n"
            
            return None, stats_text
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

def distribution_analysis(column):
    """Analyze distribution of numeric column"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    if not column:
        return None, "⚠️ Please select a column"
    
    df = app_state.current_df
    
    if column not in df.columns:
        return None, f"⚠️ Column '{column}' not found"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return None, "⚠️ Please select a numeric column"
    
    try:
        data = df[column].dropna()
        
        if len(data) < 3:
            return None, "⚠️ Not enough data points for distribution analysis"
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        ax1.hist(data, bins=min(30, len(data.unique())), density=True, alpha=0.7, 
                edgecolor='black', color='steelblue', label='Histogram')
        
        # KDE
        try:
            data.plot(kind='kde', ax=ax1, color='red', linewidth=2, label='KDE')
        except:
            pass  # Skip KDE if it fails
        
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title(f"Distribution of {column}", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()
        
        # Q-Q Plot
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Statistics
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)
        
        stats_text = f"# Distribution Analysis\n\n"
        stats_text += f"## Column: {column}\n\n"
        stats_text += f"| Statistic | Value |\n|-----------|-------|\n"
        stats_text += f"| **Count** | {len(data):,} |\n"
        stats_text += f"| **Mean** | {data.mean():.4f} |\n"
        stats_text += f"| **Median** | {data.median():.4f} |\n"
        stats_text += f"| **Mode** | {data.mode()[0] if len(data.mode()) > 0 else 'N/A'} |\n"
        stats_text += f"| **Std Dev** | {data.std():.4f} |\n"
        stats_text += f"| **Variance** | {data.var():.4f} |\n"
        stats_text += f"| **Range** | {data.max() - data.min():.4f} |\n\n"
        
        stats_text += f"## Shape Analysis\n\n"
        stats_text += f"### Skewness: {skewness:.4f}\n"
        if abs(skewness) < 0.5:
            stats_text += "- **Interpretation:** Fairly symmetric ✓\n"
        elif skewness > 0:
            stats_text += "- **Interpretation:** Right-skewed (positive skew)\n"
            stats_text += "- Tail extends to the right\n"
        else:
            stats_text += "- **Interpretation:** Left-skewed (negative skew)\n"
            stats_text += "- Tail extends to the left\n"
        
        stats_text += f"\n### Kurtosis: {kurt:.4f}\n"
        if abs(kurt) < 0.5:
            stats_text += "- **Interpretation:** Similar to normal distribution\n"
        elif kurt > 0:
            stats_text += "- **Interpretation:** Heavy-tailed (leptokurtic)\n"
            stats_text += "- More outliers than normal distribution\n"
        else:
            stats_text += "- **Interpretation:** Light-tailed (platykurtic)\n"
            stats_text += "- Fewer outliers than normal distribution\n"
        
        stats_text += f"\n## Normality Assessment\n\n"
        if abs(skewness) < 0.5 and abs(kurt) < 0.5:
            stats_text += "✓ **Distribution appears approximately normal**\n"
        else:
            stats_text += "⚠️ **Distribution deviates from normality**\n"
            stats_text += "\nConsider transformations if needed:\n"
            stats_text += "- Log transformation (for right-skewed data)\n"
            stats_text += "- Square root transformation\n"
            stats_text += "- Box-Cox transformation\n"
        
        return fig, stats_text
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

def categorical_analysis(column):
    """Analyze categorical column"""
    valid, msg = validate_dataframe()
    if not valid:
        return None, msg
    
    if not column:
        return None, "⚠️ Please select a column"
    
    df = app_state.current_df
    
    if column not in df.columns:
        return None, f"⚠️ Column '{column}' not found"
    
    try:
        value_counts = df[column].value_counts().head(15)
        
        if len(value_counts) == 0:
            return None, "⚠️ No data available in this column"
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(value_counts)))
        bars = ax1.bar(range(len(value_counts)), value_counts.values, color=colors, edgecolor='black')
        ax1.set_xticks(range(len(value_counts)))
        ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.set_title(f"Top Categories - {column}", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)
        
        # Pie chart
        ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', 
               startangle=90, colors=colors)
        ax2.set_title(f"Distribution - {column}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Statistics
        total_unique = df[column].nunique()
        total_count = df[column].count()
        missing_count = df[column].isnull().sum()
        
        stats_text = f"# Categorical Analysis\n\n"
        stats_text += f"## Column: {column}\n\n"
        stats_text += f"| Metric | Value |\n|--------|-------|\n"
        stats_text += f"| **Total Values** | {total_count:,} |\n"
        stats_text += f"| **Unique Categories** | {total_unique:,} |\n"
        stats_text += f"| **Missing Values** | {missing_count:,} |\n"
        stats_text += f"| **Most Common** | {value_counts.index[0]} |\n"
        stats_text += f"| **Mode Frequency** | {value_counts.values[0]:,} ({value_counts.values[0]/total_count*100:.1f}%) |\n\n"
        
        stats_text += f"## Top 10 Categories\n\n"
        for i, (cat, count) in enumerate(value_counts.head(10).items(), 1):
            pct = (count / total_count) * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            stats_text += f"{i}. **{cat}**: {count:,} ({pct:.1f}%) `{bar}`\n"
        
        if total_unique > 15:
            stats_text += f"\n*... and {total_unique - 15} more categories*\n"
        
        # Cardinality assessment
        stats_text += f"\n## Cardinality Assessment\n\n"
        cardinality_ratio = total_unique / total_count
        
        if cardinality_ratio > 0.9:
            stats_text += "🔴 **Very High Cardinality** - Consider feature engineering\n"
        elif cardinality_ratio > 0.5:
            stats_text += "🟡 **High Cardinality** - May need grouping\n"
        else:
            stats_text += "🟢 **Appropriate Cardinality** - Good for categorical analysis\n"
        
        return fig, stats_text
        
    except Exception as e:
        cleanup_plots()
        return None, f"❌ Error: {str(e)}"

  # ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_feature(operation, col1, col2, new_name, math_op):
    """Create new features through various operations"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not col1 or not new_name:
        return "⚠️ Please select column and provide feature name", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        if operation == "Combine (Add)":
            if not col2:
                app_state.df_history.pop()
                return "⚠️ Please select Column 2 for combination", None
            if col1 not in df.columns or col2 not in df.columns:
                app_state.df_history.pop()
                return "⚠️ One or both columns not found", None
            df[new_name] = pd.to_numeric(df[col1], errors='coerce') + pd.to_numeric(df[col2], errors='coerce')
            
        elif operation == "Combine (Multiply)":
            if not col2:
                app_state.df_history.pop()
                return "⚠️ Please select Column 2 for combination", None
            if col1 not in df.columns or col2 not in df.columns:
                app_state.df_history.pop()
                return "⚠️ One or both columns not found", None
            df[new_name] = pd.to_numeric(df[col1], errors='coerce') * pd.to_numeric(df[col2], errors='coerce')
            
        elif operation == "Combine (Divide)":
            if not col2:
                app_state.df_history.pop()
                return "⚠️ Please select Column 2 for combination", None
            if col1 not in df.columns or col2 not in df.columns:
                app_state.df_history.pop()
                return "⚠️ One or both columns not found", None
            df[new_name] = pd.to_numeric(df[col1], errors='coerce') / pd.to_numeric(df[col2], errors='coerce').replace(0, np.nan)
            
        elif operation == "Mathematical Transform":
            if col1 not in df.columns:
                app_state.df_history.pop()
                return f"⚠️ Column '{col1}' not found", None
            
            if math_op == "Square":
                df[new_name] = pd.to_numeric(df[col1], errors='coerce') ** 2
            elif math_op == "Cube":
                df[new_name] = pd.to_numeric(df[col1], errors='coerce') ** 3
            elif math_op == "Square Root":
                df[new_name] = np.sqrt(pd.to_numeric(df[col1], errors='coerce').abs())
            elif math_op == "Absolute":
                df[new_name] = pd.to_numeric(df[col1], errors='coerce').abs()
                
        elif operation == "DateTime Features":
            if col1 not in df.columns:
                app_state.df_history.pop()
                return f"⚠️ Column '{col1}' not found", None
            
            df[col1] = pd.to_datetime(df[col1], errors='coerce')
            df[f"{col1}_year"] = df[col1].dt.year
            df[f"{col1}_month"] = df[col1].dt.month
            df[f"{col1}_day"] = df[col1].dt.day
            df[f"{col1}_dayofweek"] = df[col1].dt.dayofweek
            df[f"{col1}_quarter"] = df[col1].dt.quarter
            
            app_state.current_df = df
            return "✓ DateTime features created (year, month, day, dayofweek, quarter)", df.head(20)
        
        app_state.current_df = df
        return f"✓ Feature '{new_name}' created successfully using {operation}", df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def transform_features(columns, method):
    """Transform features using various methods"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        from sklearn.preprocessing import PowerTransformer
        
        processed = []
        skipped = []
        
        for col in columns:
            if col not in df.columns:
                skipped.append(f"{col} (not found)")
                continue
            
            if not pd.api.types.is_numeric_dtype(df[col]):
                skipped.append(f"{col} (not numeric)")
                continue
            
            if method == "Log Transform":
                df[col] = np.log1p(df[col].clip(lower=0))
            elif method == "Square Transform":
                df[col] = df[col] ** 2
            elif method == "Power Transform (Yeo-Johnson)":
                pt = PowerTransformer(method='yeo-johnson')
                df[col] = pt.fit_transform(df[[col]])
            
            processed.append(col)
        
        app_state.current_df = df
        
        status = f"✓ Transformation applied: {method}\n"
        if processed:
            status += f"✓ Processed: {', '.join(processed)}"
        if skipped:
            status += f"\n⚠️ Skipped: {', '.join(skipped)}"
        
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def encode_features(columns, method):
    """Encode categorical features"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        
        if method == "Label Encoding":
            for col in columns:
                if col not in df.columns:
                    continue
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        elif method == "One-Hot Encoding":
            valid_cols = [col for col in columns if col in df.columns]
            if not valid_cols:
                app_state.df_history.pop()
                return "⚠️ No valid columns found", None
            df = pd.get_dummies(df, columns=valid_cols, prefix=valid_cols, drop_first=False)
        
        elif method == "Ordinal Encoding":
            valid_cols = [col for col in columns if col in df.columns]
            if not valid_cols:
                app_state.df_history.pop()
                return "⚠️ No valid columns found", None
            oe = OrdinalEncoder()
            df[valid_cols] = oe.fit_transform(df[valid_cols].astype(str))
        
        app_state.current_df = df
        
        new_cols = len(df.columns) - len(app_state.df_history[-1].columns)
        status = f"✓ Encoding applied: {method}\n"
        status += f"✓ Processed {len(columns)} column(s)"
        if new_cols > 0:
            status += f"\n📊 Created {new_cols} new columns"
        
        return status, df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def apply_binning(column, method, n_bins):
    """Apply binning/discretization to numeric column"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not column:
        return "⚠️ Please select a column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        if column not in df.columns:
            app_state.df_history.pop()
            return f"⚠️ Column '{column}' not found", None
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            app_state.df_history.pop()
            return "⚠️ Binning requires numeric columns", None
        
        n_bins = int(n_bins)
        
        if method == "Equal Width":
            df[f"{column}_binned"] = pd.cut(df[column], bins=n_bins, labels=False, duplicates='drop')
        elif method == "Equal Frequency":
            df[f"{column}_binned"] = pd.qcut(df[column], q=n_bins, labels=False, duplicates='drop')
        
        app_state.current_df = df
        
        unique_bins = df[f"{column}_binned"].nunique()
        return f"✓ Binning applied: {method} with {unique_bins} bins\n✓ New column: {column}_binned", df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

def reduce_dimensions(method, n_components):
    """Apply dimensionality reduction"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None, None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        from sklearn.decomposition import PCA
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            app_state.df_history.pop()
            return "⚠️ No numeric columns found", None, None
        
        n_components = int(n_components)
        
        if n_components > len(numeric_df.columns):
            app_state.df_history.pop()
            return f"⚠️ Cannot create {n_components} components from {len(numeric_df.columns)} features", None, None
        
        if method == "PCA (Principal Component Analysis)":
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(numeric_df.fillna(0))
            
            # Add PCA components to dataframe
            for i in range(n_components):
                df[f'PC{i+1}'] = components[:, i]
            
            # Plot explained variance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
                  edgecolor='black', color='steelblue', alpha=0.7)
            ax.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio()), 
                   'r-o', linewidth=2, markersize=8, label='Cumulative')
            ax.set_xlabel('Principal Component', fontsize=12)
            ax.set_ylabel('Explained Variance Ratio', fontsize=12)
            ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
            plt.tight_layout()
            
            app_state.current_df = df
            total_var = pca.explained_variance_ratio_.sum()
            
            status = f"✓ PCA applied: {n_components} components\n"
            status += f"📊 Explains {total_var:.2%} of total variance\n"
            status += f"✓ New columns: PC1 to PC{n_components}"
            
            return status, df.head(20), fig
    
    except Exception as e:
        app_state.df_history.pop()
        cleanup_plots()
        return f"❌ Error: {str(e)}", None, None

def create_polynomial(columns, degree):
    """Create polynomial features"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None
    
    if not columns:
        return "⚠️ Please select at least one column", None
    
    app_state.save_to_history()
    df = app_state.current_df.copy()
    
    try:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Validate columns
        valid_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not valid_cols:
            app_state.df_history.pop()
            return "⚠️ No valid numeric columns selected", None
        
        degree = int(degree)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[valid_cols].fillna(0))
        
        # Get feature names
        feature_names = poly.get_feature_names_out(valid_cols)
        
        # Add polynomial features (skip original features)
        for i, name in enumerate(feature_names):
            if name not in valid_cols:
                df[name] = poly_features[:, i]
        
        app_state.current_df = df
        new_features = len(feature_names) - len(valid_cols)
        
        return f"✓ Created {new_features} polynomial features (degree {degree})", df.head(20)
        
    except Exception as e:
        app_state.df_history.pop()
        return f"❌ Error: {str(e)}", None

  # ============================================================================
# MODEL BUILDING FUNCTIONS - Part 1
# ============================================================================

def get_target_info():
    """Get information about potential target variables"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg
    
    df = app_state.current_df
    info = ["# 📋 Target Variable Selection Guide\n"]
    info.append(f"**Total Columns:** {len(df.columns)}\n")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    info.append(f"## 🔢 Numeric Columns ({len(numeric_cols)})")
    info.append("*Suitable for Regression tasks*\n")
    
    for col in numeric_cols[:10]:
        unique_count = df[col].nunique()
        min_val = df[col].min()
        max_val = df[col].max()
        info.append(f"- **{col}**: {unique_count} unique values, range [{min_val:.2f}, {max_val:.2f}]")
    
    if len(numeric_cols) > 10:
        info.append(f"- *... and {len(numeric_cols) - 10} more*\n")
    
    info.append(f"\n## 📝 Categorical Columns ({len(categorical_cols)})")
    info.append("*Suitable for Classification tasks*\n")
    
    for col in categorical_cols[:10]:
        unique_count = df[col].nunique()
        top_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
        info.append(f"- **{col}**: {unique_count} unique values, most common: '{top_val}'")
    
    if len(categorical_cols) > 10:
        info.append(f"- *... and {len(categorical_cols) - 10} more*\n")
    
    info.append(f"\n## 💡 Recommendations\n")
    info.append(f"- **Classification**: Choose categorical column with 2-20 unique classes")
    info.append(f"- **Regression**: Choose numeric column with continuous values")
    info.append(f"- **Clustering**: No target needed (unsupervised learning)")
    
    return "\n".join(info)

def prepare_data_split(target_col, task_type, test_size, val_size, use_validation):
    """Prepare train-test split for model building"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None, None, None, None
    
    if not target_col:
        return "⚠️ Please select a target variable", None, None, None, None
    
    df = app_state.current_df
    
    if target_col not in df.columns:
        return f"⚠️ Target column '{target_col}' not found", None, None, None, None
    
    try:
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Select only numeric features
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return "⚠️ No numeric features found. Please ensure you have numeric columns for modeling.", None, None, None, None
        
        # Remove rows with missing target
        mask = ~y.isnull()
        X_numeric = X_numeric[mask]
        y = y[mask]
        
        if len(X_numeric) == 0:
            return "⚠️ No data remaining after removing missing target values", None, None, None, None
        
        # Fill remaining missing values in features
        X_numeric = X_numeric.fillna(X_numeric.mean())
        
        # Store in app state
        app_state.model_type = task_type
        
        # Perform train-test split
        app_state.X_train, app_state.X_test, app_state.y_train, app_state.y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y if task_type == "Classification" and y.nunique() > 1 and y.nunique() < 20 else None
        )
        
        info = f"# ✓ Data Split Complete\n\n"
        info += f"| Parameter | Value |\n|-----------|-------|\n"
        info += f"| **Task Type** | {task_type} |\n"
        info += f"| **Target Variable** | {target_col} |\n"
        info += f"| **Features Used** | {X_numeric.shape[1]} |\n"
        info += f"| **Training Set** | {app_state.X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%) |\n"
        info += f"| **Test Set** | {app_state.X_test.shape[0]:,} samples ({test_size*100:.0f}%) |\n"
        
        if task_type == "Classification":
            train_dist = app_state.y_train.value_counts()
            info += f"\n## Training Set Class Distribution\n\n"
            for cls, count in train_dist.items():
                pct = (count / len(app_state.y_train)) * 100
                info += f"- **{cls}**: {count:,} ({pct:.1f}%)\n"
        
        if use_validation:
            val_ratio = val_size / (1 - test_size)
            app_state.X_train, X_val, app_state.y_train, y_val = train_test_split(
                app_state.X_train, app_state.y_train, test_size=val_ratio, random_state=42
            )
            info += f"| **Validation Set** | {X_val.shape[0]:,} samples |\n"
        
        info += f"\n## Feature Names\n\n"
        feature_list = app_state.X_train.columns.tolist()
        info += f"*{', '.join(feature_list[:10])}*"
        if len(feature_list) > 10:
            info += f" *... and {len(feature_list)-10} more*"
        
        return info, app_state.X_train.head(), app_state.X_test.head(), None, None
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None, None

def perform_feature_selection(method, n_features):
    """Perform feature selection"""
    if app_state.X_train is None or app_state.y_train is None:
        return "⚠️ Please split data first", None
    
    try:
        n_features = min(int(n_features), len(app_state.X_train.columns))
        
        if method == "Correlation with Target":
            # For numeric target
            if pd.api.types.is_numeric_dtype(app_state.y_train):
                correlations = pd.DataFrame(app_state.X_train).corrwith(pd.Series(app_state.y_train)).abs().sort_values(ascending=False)
                selected_features = correlations.head(n_features).index.tolist()
                
                info = f"# Feature Selection Results\n\n"
                info += f"**Method:** {method}\n"
                info += f"**Selected Features:** {len(selected_features)}\n\n"
                info += f"## Top {n_features} Features\n\n"
                info += f"| Rank | Feature | Correlation |\n|------|---------|-------------|\n"
                for i, (feat_idx, corr) in enumerate(correlations.head(n_features).items(), 1):
                    feat_name = app_state.X_train.columns[feat_idx]
                    info += f"| {i} | **{feat_name}** | {corr:.4f} |\n"
                
                return info, selected_features
            else:
                return "⚠️ This method requires numeric target variable", None
        
        elif method == "F-Test (ANOVA)":
            selector = SelectKBest(
                f_classif if app_state.model_type == "Classification" else f_regression, 
                k=n_features
            )
            selector.fit(app_state.X_train, app_state.y_train)
            selected_features = app_state.X_train.columns[selector.get_support()].tolist()
            scores = selector.scores_[selector.get_support()]
            
            info = f"# Feature Selection Results\n\n"
            info += f"**Method:** {method}\n"
            info += f"**Selected Features:** {len(selected_features)}\n\n"
            info += f"## Top Features\n\n"
            info += f"| Rank | Feature | F-Score |\n|------|---------|----------|\n"
            for i, (feat, score) in enumerate(zip(selected_features, scores), 1):
                info += f"| {i} | **{feat}** | {score:.2f} |\n"
            
            return info, selected_features
        
        elif method == "Recursive Feature Elimination (RFE)":
            estimator = (RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) 
                        if app_state.model_type == "Classification" 
                        else RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
            
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            selector.fit(app_state.X_train, app_state.y_train)
            selected_features = app_state.X_train.columns[selector.get_support()].tolist()
            rankings = selector.ranking_[selector.get_support()]
            
            info = f"# Feature Selection Results\n\n"
            info += f"**Method:** {method}\n"
            info += f"**Selected Features:** {len(selected_features)}\n\n"
            info += f"## Selected Features\n\n"
            for i, feat in enumerate(selected_features, 1):
                info += f"{i}. **{feat}**\n"
            
            return info, selected_features
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

  # ============================================================================
# MODEL BUILDING FUNCTIONS - Part 2 (Training and Evaluation)
# ============================================================================

def train_model(algorithm):
    """Train machine learning model"""
    if app_state.X_train is None or app_state.y_train is None:
        return "⚠️ Please split data first", None, "Error"
    
    try:
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0, max_iter=10000),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree (Classification)": DecisionTreeClassifier(random_state=42, max_depth=10),
            "Decision Tree (Regression)": DecisionTreeRegressor(random_state=42, max_depth=10),
            "Random Forest (Classification)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Random Forest (Regression)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting (Classification)": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting (Regression)": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "K-Nearest Neighbors (Classification)": KNeighborsClassifier(n_neighbors=5),
            "K-Nearest Neighbors (Regression)": KNeighborsRegressor(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Support Vector Machine (Classification)": SVC(probability=True, random_state=42),
            "Support Vector Machine (Regression)": SVR()
        }
        
        model = models.get(algorithm)
        if model is None:
            return "⚠️ Invalid algorithm selected", None, "Error"
        
        # Train model
        model.fit(app_state.X_train, app_state.y_train)
        app_state.trained_model = model
        
        # Make predictions
        y_pred_train = model.predict(app_state.X_train)
        y_pred_test = model.predict(app_state.X_test)
        
        info = f"# ✓ Model Training Complete\n\n"
        info += f"| Parameter | Value |\n|-----------|-------|\n"
        info += f"| **Algorithm** | {algorithm} |\n"
        info += f"| **Training Samples** | {len(app_state.X_train):,} |\n"
        info += f"| **Test Samples** | {len(app_state.X_test):,} |\n"
        info += f"| **Features** | {app_state.X_train.shape[1]} |\n"
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': app_state.X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            app_state.feature_importance_data = importance
            
            info += f"\n## 🎯 Top 10 Important Features\n\n"
            info += f"| Rank | Feature | Importance |\n|------|---------|------------|\n"
            for i, (idx, row) in enumerate(importance.head(10).iterrows(), 1):
                info += f"| {i} | **{row['feature']}** | {row['importance']:.4f} |\n"
        
        # Create visualization
        fig = None
        
        if app_state.model_type == "Classification":
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ConfusionMatrixDisplay.from_predictions(
                app_state.y_train, y_pred_train, ax=ax1, cmap='Blues', colorbar=False
            )
            ax1.set_title('Training Set - Confusion Matrix', fontsize=12, fontweight='bold')
            ax1.grid(False)
            
            ConfusionMatrixDisplay.from_predictions(
                app_state.y_test, y_pred_test, ax=ax2, cmap='Blues', colorbar=False
            )
            ax2.set_title('Test Set - Confusion Matrix', fontsize=12, fontweight='bold')
            ax2.grid(False)
            
            plt.tight_layout()
        
        elif app_state.model_type == "Regression":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Training set
            ax1.scatter(app_state.y_train, y_pred_train, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax1.plot([app_state.y_train.min(), app_state.y_train.max()], 
                    [app_state.y_train.min(), app_state.y_train.max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('Actual Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title('Training: Actual vs Predicted', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend()
            
            # Test set
            ax2.scatter(app_state.y_test, y_pred_test, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            ax2.plot([app_state.y_test.min(), app_state.y_test.max()], 
                    [app_state.y_test.min(), app_state.y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
            ax2.set_xlabel('Actual Values', fontsize=12)
            ax2.set_ylabel('Predicted Values', fontsize=12)
            ax2.set_title('Test: Actual vs Predicted', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend()
            
            plt.tight_layout()
        
        return info, fig, "✓ Model trained successfully!"
        
    except Exception as e:
        cleanup_plots()
        return f"❌ Error: {str(e)}", None, "Training failed"

def evaluate_model():
    """Evaluate trained model"""
    if app_state.trained_model is None:
        return "⚠️ Please train a model first", None
    
    try:
        y_pred_train = app_state.trained_model.predict(app_state.X_train)
        y_pred_test = app_state.trained_model.predict(app_state.X_test)
        
        info = f"# 📊 Model Evaluation Results\n\n"
        
        if app_state.model_type == "Classification":
            train_acc = accuracy_score(app_state.y_train, y_pred_train)
            test_acc = accuracy_score(app_state.y_test, y_pred_test)
            train_prec = precision_score(app_state.y_train, y_pred_train, average='weighted', zero_division=0)
            test_prec = precision_score(app_state.y_test, y_pred_test, average='weighted', zero_division=0)
            train_rec = recall_score(app_state.y_train, y_pred_train, average='weighted', zero_division=0)
            test_rec = recall_score(app_state.y_test, y_pred_test, average='weighted', zero_division=0)
            train_f1 = f1_score(app_state.y_train, y_pred_train, average='weighted', zero_division=0)
            test_f1 = f1_score(app_state.y_test, y_pred_test, average='weighted', zero_division=0)
            
            info += f"## Classification Metrics\n\n"
            info += f"| Metric | Training | Test | Difference |\n"
            info += f"|--------|----------|------|------------|\n"
            info += f"| **Accuracy** | {train_acc:.4f} | {test_acc:.4f} | {train_acc-test_acc:+.4f} |\n"
            info += f"| **Precision** | {train_prec:.4f} | {test_prec:.4f} | {train_prec-test_prec:+.4f} |\n"
            info += f"| **Recall** | {train_rec:.4f} | {test_rec:.4f} | {train_rec-test_rec:+.4f} |\n"
            info += f"| **F1-Score** | {train_f1:.4f} | {test_f1:.4f} | {train_f1-test_f1:+.4f} |\n\n"
            
            # ROC-AUC for binary classification
            if len(np.unique(app_state.y_train)) == 2:
                try:
                    if hasattr(app_state.trained_model, 'predict_proba'):
                        y_pred_proba = app_state.trained_model.predict_proba(app_state.X_test)[:, 1]
                        roc_auc = roc_auc_score(app_state.y_test, y_pred_proba)
                        info += f"**ROC-AUC Score:** {roc_auc:.4f}\n\n"
                except:
                    pass
            
            info += f"## Classification Report (Test Set)\n\n```\n"
            info += classification_report(app_state.y_test, y_pred_test)
            info += "```\n\n"
            
            # Model diagnosis
            info += f"## 🔍 Model Diagnosis\n\n"
            diff = train_acc - test_acc
            if diff > 0.1:
                info += f"⚠️ **Warning: Possible Overfitting**\n"
                info += f"- Training accuracy significantly higher than test accuracy\n"
                info += f"- Consider: regularization, reduce model complexity, or get more data\n"
            elif diff < -0.05:
                info += f"⚠️ **Warning: Possible Underfitting**\n"
                info += f"- Test accuracy higher than training accuracy (unusual)\n"
                info += f"- Consider: checking data quality or model appropriateness\n"
            elif test_acc < 0.6:
                info += f"⚠️ **Warning: Low Performance**\n"
                info += f"- Model accuracy below 60%\n"
                info += f"- Consider: feature engineering, different algorithm, or more data\n"
            else:
                info += f"✓ **Good: Balanced Performance**\n"
                info += f"- Model generalizes well to unseen data\n"
        
        elif app_state.model_type == "Regression":
            train_mae = mean_absolute_error(app_state.y_train, y_pred_train)
            test_mae = mean_absolute_error(app_state.y_test, y_pred_test)
            train_mse = mean_squared_error(app_state.y_train, y_pred_train)
            test_mse = mean_squared_error(app_state.y_test, y_pred_test)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_r2 = r2_score(app_state.y_train, y_pred_train)
            test_r2 = r2_score(app_state.y_test, y_pred_test)
            
            info += f"## Regression Metrics\n\n"
            info += f"| Metric | Training | Test | Difference |\n"
            info += f"|--------|----------|------|------------|\n"
            info += f"| **MAE** | {train_mae:.4f} | {test_mae:.4f} | {train_mae-test_mae:+.4f} |\n"
            info += f"| **MSE** | {train_mse:.4f} | {test_mse:.4f} | {train_mse-test_mse:+.4f} |\n"
            info += f"| **RMSE** | {train_rmse:.4f} | {test_rmse:.4f} | {train_rmse-test_rmse:+.4f} |\n"
            info += f"| **R² Score** | {train_r2:.4f} | {test_r2:.4f} | {train_r2-test_r2:+.4f} |\n\n"
            
            # Model diagnosis
            info += f"## 🔍 Model Diagnosis\n\n"
            diff = train_r2 - test_r2
            if diff > 0.15:
                info += f"⚠️ **Warning: Possible Overfitting**\n"
                info += f"- Training R² significantly higher than test R²\n"
                info += f"- Consider: regularization or reduce model complexity\n"
            elif test_r2 < 0.3:
                info += f"⚠️ **Warning: Low R² Score**\n"
                info += f"- Model explains less than 30% of variance\n"
                info += f"- Consider: feature engineering or different algorithm\n"
            elif test_r2 < 0.5:
                info += f"🟡 **Moderate Performance**\n"
                info += f"- Model explains {test_r2*100:.1f}% of variance\n"
                info += f"- Room for improvement through feature engineering\n"
            else:
                info += f"✓ **Good Performance**\n"
                info += f"- Model explains {test_r2*100:.1f}% of variance\n"
        
        # Feature importance plot
        fig = None
        if app_state.feature_importance_data is not None:
            fig, ax = plt.subplots(figsize=(10, max(6, len(app_state.feature_importance_data.head(15)) * 0.4)))
            top_features = app_state.feature_importance_data.head(15)
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
            bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors, edgecolor='black')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Feature Importance (Top 15)', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.4f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
        
        return info, fig
        
    except Exception as e:
        cleanup_plots()
        return f"❌ Error: {str(e)}", None

  # ============================================================================
# MODEL BUILDING FUNCTIONS - Part 3 (Cross-validation, Tuning, Clustering)
# ============================================================================

def perform_cross_validation(k_folds):
    """Perform k-fold cross-validation"""
    if app_state.trained_model is None:
        return "⚠️ Please train a model first"
    
    try:
        k_folds = int(k_folds)
        
        if k_folds < 2:
            return "⚠️ Number of folds must be at least 2"
        
        if k_folds > len(app_state.X_train):
            return f"⚠️ Number of folds ({k_folds}) cannot exceed training samples ({len(app_state.X_train)})"
        
        scoring = 'accuracy' if app_state.model_type == "Classification" else 'r2'
        scores = cross_val_score(app_state.trained_model, app_state.X_train, app_state.y_train, 
                                cv=k_folds, scoring=scoring, n_jobs=-1)
        
        info = f"# Cross-Validation Results\n\n"
        info += f"| Parameter | Value |\n|-----------|-------|\n"
        info += f"| **K-Folds** | {k_folds} |\n"
        info += f"| **Scoring Metric** | {scoring} |\n"
        info += f"| **Mean Score** | {scores.mean():.4f} |\n"
        info += f"| **Std Dev** | {scores.std():.4f} |\n"
        info += f"| **Min Score** | {scores.min():.4f} |\n"
        info += f"| **Max Score** | {scores.max():.4f} |\n\n"
        
        info += f"## Individual Fold Scores\n\n"
        info += f"| Fold | Score |\n|------|-------|\n"
        for i, score in enumerate(scores, 1):
            info += f"| {i} | {score:.4f} |\n"
        
        info += f"\n## Interpretation\n\n"
        if scores.std() < 0.05:
            info += f"✓ **Stable Model**: Low variance across folds indicates consistent performance\n"
        else:
            info += f"⚠️ **Variable Performance**: High variance suggests model sensitivity to data splits\n"
        
        return info
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def hyperparameter_tuning(algorithm, search_method, n_iter):
    """Perform hyperparameter tuning"""
    if app_state.X_train is None or app_state.y_train is None:
        return "⚠️ Please split data first", None
    
    try:
        n_iter = int(n_iter)
        
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "Gradient Boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            "Logistic Regression": {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
        
        # Find matching param grid
        param_grid = None
        model = None
        
        for key in param_grids:
            if key in algorithm:
                param_grid = param_grids[key]
                if "Random Forest" in algorithm:
                    model = (RandomForestClassifier(random_state=42, n_jobs=-1) 
                            if app_state.model_type == "Classification" 
                            else RandomForestRegressor(random_state=42, n_jobs=-1))
                elif "Gradient Boosting" in algorithm:
                    model = (GradientBoostingClassifier(random_state=42) 
                            if app_state.model_type == "Classification" 
                            else GradientBoostingRegressor(random_state=42))
                elif "Logistic Regression" in algorithm:
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif "K-Nearest Neighbors" in algorithm:
                    model = (KNeighborsClassifier() 
                            if app_state.model_type == "Classification" 
                            else KNeighborsRegressor())
                break
        
        if param_grid is None or model is None:
            return f"⚠️ Hyperparameter tuning not configured for {algorithm}", None
        
        # Perform search
        if search_method == "Grid Search":
            search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=0)
        else:
            search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=5, 
                                       random_state=42, n_jobs=-1, verbose=0)
        
        search.fit(app_state.X_train, app_state.y_train)
        app_state.trained_model = search.best_estimator_
        
        info = f"# Hyperparameter Tuning Results\n\n"
        info += f"| Parameter | Value |\n|-----------|-------|\n"
        info += f"| **Algorithm** | {algorithm} |\n"
        info += f"| **Search Method** | {search_method} |\n"
        info += f"| **CV Folds** | 5 |\n"
        info += f"| **Best CV Score** | {search.best_score_:.4f} |\n\n"
        
        info += f"## Best Parameters\n\n"
        for param, value in search.best_params_.items():
            info += f"- **{param}**: {value}\n"
        
        info += f"\n## Top 5 Parameter Combinations\n\n"
        results_df = pd.DataFrame(search.cv_results_).sort_values('mean_test_score', ascending=False).head(5)
        
        for idx, row in results_df.iterrows():
            rank = results_df.index.get_loc(idx) + 1
            info += f"### Rank {rank} (Score: {row['mean_test_score']:.4f})\n"
            for k, v in row['params'].items():
                info += f"- **{k}**: {v}\n"
            info += "\n"
        
        return info, "✓ Model updated with best parameters!"
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def handle_imbalanced_data(method):
    """Handle imbalanced datasets"""
    if app_state.X_train is None or app_state.y_train is None:
        return "⚠️ Please split data first", None, None
    
    if app_state.model_type != "Classification":
        return "⚠️ This feature is only for classification tasks", None, None
    
    try:
        class_dist = pd.Series(app_state.y_train).value_counts()
        
        info = f"# Class Distribution Analysis\n\n"
        info += f"## Before Balancing\n\n"
        info += f"| Class | Count | Percentage |\n|-------|-------|------------|\n"
        
        for cls, count in class_dist.items():
            pct = (count / len(app_state.y_train)) * 100
            info += f"| {cls} | {count:,} | {pct:.1f}% |\n"
        
        # Check imbalance ratio
        max_class = class_dist.max()
        min_class = class_dist.min()
        imbalance_ratio = max_class / min_class
        
        info += f"\n**Imbalance Ratio:** {imbalance_ratio:.2f}:1\n\n"
        
        if imbalance_ratio < 1.5:
            info += f"✓ **Classes are balanced** - No action needed\n"
            return info, app_state.X_train, app_state.y_train
        
        if method == "SMOTE (Oversampling)":
            smote = SMOTE(random_state=42)
            X_train_new, y_train_new = smote.fit_resample(app_state.X_train, app_state.y_train)
            
            class_dist_new = pd.Series(y_train_new).value_counts()
            
            info += f"## After SMOTE\n\n"
            info += f"| Class | Count | Percentage |\n|-------|-------|------------|\n"
            
            for cls, count in class_dist_new.items():
                pct = (count / len(y_train_new)) * 100
                info += f"| {cls} | {count:,} | {pct:.1f}% |\n"
            
            info += f"\n**Total Samples:** {len(app_state.y_train):,} → {len(y_train_new):,} "
            info += f"(+{len(y_train_new) - len(app_state.y_train):,} synthetic samples)\n\n"
            info += f"✓ **Classes are now balanced**\n"
            
            return info, X_train_new, y_train_new
        
        elif method == "Class Weights":
            info += f"## Recommendation for Class Weights\n\n"
            info += f"When training your model, use the parameter:\n\n"
            info += f"```python\n"
            info += f"class_weight='balanced'\n"
            info += f"```\n\n"
            info += f"This automatically adjusts weights inversely proportional to class frequencies.\n\n"
            info += f"**Calculated Weights:**\n\n"
            
            n_samples = len(app_state.y_train)
            n_classes = len(class_dist)
            
            for cls, count in class_dist.items():
                weight = n_samples / (n_classes * count)
                info += f"- Class {cls}: {weight:.2f}\n"
            
            return info, app_state.X_train, app_state.y_train
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def perform_clustering(algorithm, n_clusters):
    """Perform clustering analysis"""
    valid, msg = validate_dataframe()
    if not valid:
        return msg, None, None
    
    try:
        numeric_df = app_state.current_df.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_df.columns) < 2:
            return "⚠️ Need at least 2 numeric columns for clustering", None, None
        
        n_clusters = int(n_clusters)
        
        # Choose algorithm
        if algorithm == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == "Hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return "⚠️ Invalid algorithm", None, None
        
        # Fit model
        labels = model.fit_predict(numeric_df)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calculate metrics
        if n_clusters_found > 1 and -1 not in labels:
            silhouette = silhouette_score(numeric_df, labels)
            davies_bouldin = davies_bouldin_score(numeric_df, labels)
        else:
            silhouette = None
            davies_bouldin = None
        
        info = f"# Clustering Results\n\n"
        info += f"| Parameter | Value |\n|-----------|-------|\n"
        info += f"| **Algorithm** | {algorithm} |\n"
        info += f"| **Clusters Found** | {n_clusters_found} |\n"
        info += f"| **Data Points** | {len(numeric_df):,} |\n"
        info += f"| **Features** | {len(numeric_df.columns)} |\n"
        
        if silhouette is not None:
            info += f"| **Silhouette Score** | {silhouette:.4f} |\n"
            info += f"| **Davies-Bouldin** | {davies_bouldin:.4f} |\n"
        
        info += f"\n## Quality Metrics Interpretation\n\n"
        if silhouette is not None:
            info += f"**Silhouette Score** ({silhouette:.4f}):\n"
            if silhouette > 0.7:
                info += "- ✓ Excellent clustering structure\n"
            elif silhouette > 0.5:
                info += "- ✓ Good clustering structure\n"
            elif silhouette > 0.25:
                info += "- 🟡 Moderate clustering structure\n"
            else:
                info += "- ⚠️ Weak clustering structure\n"
            
            info += f"\n**Davies-Bouldin Index** ({davies_bouldin:.4f}):\n"
            if davies_bouldin < 0.5:
                info += "- ✓ Excellent separation\n"
            elif davies_bouldin < 1.0:
                info += "- ✓ Good separation\n"
            else:
                info += "- ⚠️ Poor separation\n"
        
        info += f"\n## Cluster Distribution\n\n"
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        
        info += f"| Cluster | Count | Percentage |\n|---------|-------|------------|\n"
        for cluster, count in cluster_counts.items():
            pct = (count / len(labels)) * 100
            cluster_name = "Noise" if cluster == -1 else str(cluster)
            info += f"| {cluster_name} | {count:,} | {pct:.1f}% |\n"
        
        # Visualization
        from sklearn.decomposition import PCA
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if numeric_df.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(numeric_df)
            info += f"\n*Visualization uses PCA to reduce to 2D (explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)*\n"
        else:
            coords = numeric_df.values
        
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', 
                           alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'{algorithm} Clustering (n={n_clusters_found} clusters)', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return info, fig, "✓ Clustering complete!"
        
    except Exception as e:
        cleanup_plots()
        return f"❌ Error: {str(e)}", None, None

def save_trained_model(filename):
    """Save trained model to file"""
    if app_state.trained_model is None:
        return "⚠️ No trained model to save"
    
    try:
        if not filename:
            filename = "trained_model"
        
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        joblib.dump(app_state.trained_model, filename)
        
        # Also save feature names
        if app_state.X_train is not None:
            feature_names = app_state.X_train.columns.tolist()
            with open(filename.replace('.pkl', '_features.txt'), 'w') as f:
                f.write('\n'.join(feature_names))
        
        return f"✓ Model saved successfully!\n\n- Model: {filename}\n- Features: {filename.replace('.pkl', '_features.txt')}"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# GRADIO UI DEFINITION - Part 1 (Upload and Data Cleaning Tabs)
# ============================================================================

with gr.Blocks(title="Advanced EDA & ML App", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📊 Advanced EDA & Machine Learning Application")
    gr.Markdown("**Upload CSV, Excel, or JSON files for comprehensive data analysis and machine learning**")
    
    # ========================================================================
    # TAB 1: UPLOAD & OVERVIEW
    # ========================================================================
    
    with gr.Tab("📁 Upload & Overview"):
        with gr.Row():
            file_input = gr.File(
                label="Upload Dataset (CSV, Excel, JSON)",
                file_types=[".csv", ".xlsx", ".xls", ".json"]
            )
        
        with gr.Row():
            info_output = gr.Textbox(label="Dataset Info", lines=2, interactive=False)
        
        table_output = gr.Dataframe(label="Data Preview (First 20 rows)", wrap=True, height=400)
    
    # ========================================================================
    # TAB 2: DATA CLEANING
    # ========================================================================
    
    with gr.Tab("🧹 Data Cleaning"):
        gr.Markdown("### Data Quality Dashboard")
        
        with gr.Row():
            summary_btn = gr.Button("📊 Generate Quality Report", variant="primary", size="lg")
            undo_btn = gr.Button("↩️ Undo Last Action", variant="secondary")
            reset_btn = gr.Button("🔄 Reset Dataset", variant="stop")
        
        summary_output = gr.Markdown()
        
        with gr.Row():
            undo_status = gr.Textbox(label="Status", visible=False)
            undo_preview = gr.Dataframe(label="Preview After Undo", visible=False)
        
        # Connect buttons
        summary_btn.click(fn=get_cleaning_summary, outputs=summary_output)
        
        undo_btn.click(
            fn=undo_last_action,
            outputs=[undo_status, undo_preview]
        ).then(
            lambda: (gr.update(visible=True), gr.update(visible=True)),
            outputs=[undo_status, undo_preview]
        )
        
        reset_btn.click(fn=reset_data, outputs=[undo_status, undo_preview])
        
        gr.Markdown("---")
        
        # 1. Missing Values
        with gr.Accordion("1️⃣ Handle Missing Values", open=False):
            with gr.Row():
                missing_strategy = gr.Dropdown(
                    choices=["Drop rows", "Fill with mean", "Fill with median", 
                            "Fill with mode", "Fill with custom value", 
                            "Forward fill", "Backward fill"],
                    label="Strategy",
                    value="Drop rows"
                )
                missing_cols = gr.Dropdown(
                    choices=[],
                    label="Select Columns",
                    multiselect=True,
                    interactive=True
                )
                fill_val = gr.Textbox(label="Custom Fill Value", value="0", 
                                     info="Used only with 'Fill with custom value'")
            
            with gr.Row():
                missing_btn = gr.Button("Apply", variant="primary")
                missing_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            missing_status = gr.Textbox(label="Status", interactive=False)
            missing_preview = gr.Dataframe(label="Preview", height=300)
            
            missing_btn.click(
                fn=handle_missing_values,
                inputs=[missing_strategy, missing_cols, fill_val],
                outputs=[missing_status, missing_preview]
            )
            missing_undo_btn.click(fn=undo_last_action, outputs=[missing_status, missing_preview])
        
        # 2. Remove Duplicates
        with gr.Accordion("2️⃣ Remove Duplicates", open=False):
            dup_cols = gr.Dropdown(
                choices=[],
                label="Select Subset Columns (leave empty for all columns)",
                multiselect=True,
                interactive=True,
                info="Check for duplicates based on these columns only"
            )
            
            with gr.Row():
                dup_btn = gr.Button("Remove Duplicates", variant="primary")
                dup_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dup_status = gr.Textbox(label="Status", interactive=False)
            dup_preview = gr.Dataframe(label="Preview", height=300)
            
            dup_btn.click(fn=remove_duplicates, inputs=dup_cols, outputs=[dup_status, dup_preview])
            dup_undo_btn.click(fn=undo_last_action, outputs=[dup_status, dup_preview])
        
        # 3. Handle Outliers
        with gr.Accordion("3️⃣ Handle Outliers", open=False):
            with gr.Row():
                outlier_method = gr.Dropdown(
                    choices=["IQR Method", "Z-Score Method", "Cap outliers"],
                    label="Method",
                    value="IQR Method",
                    info="IQR: removes outliers | Z-Score: statistical | Cap: limits values"
                )
                outlier_cols = gr.Dropdown(
                    choices=[],
                    label="Select Numeric Columns",
                    multiselect=True,
                    interactive=True
                )
                z_threshold = gr.Number(label="Z-Score Threshold", value=3,
                                       info="Used only with Z-Score method")
            
            with gr.Row():
                outlier_btn = gr.Button("Apply", variant="primary")
                outlier_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            outlier_status = gr.Textbox(label="Status", interactive=False)
            outlier_preview = gr.Dataframe(label="Preview", height=300)
            
            outlier_btn.click(
                fn=handle_outliers,
                inputs=[outlier_method, outlier_cols, z_threshold],
                outputs=[outlier_status, outlier_preview]
            )
            outlier_undo_btn.click(fn=undo_last_action, outputs=[outlier_status, outlier_preview])
        
        # 4. Correct Data Types
        with gr.Accordion("4️⃣ Correct Data Types", open=False):
            with gr.Row():
                dtype_col = gr.Dropdown(choices=[], label="Select Column", interactive=True)
                dtype_type = gr.Dropdown(
                    choices=["int", "float", "string", "datetime", "category", "bool"],
                    label="New Type",
                    value="int"
                )
            
            with gr.Row():
                dtype_btn = gr.Button("Convert", variant="primary")
                dtype_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dtype_status = gr.Textbox(label="Status", interactive=False)
            dtype_preview = gr.Dataframe(label="Preview", height=300)
            
            dtype_btn.click(
                fn=correct_data_types,
                inputs=[dtype_col, dtype_type],
                outputs=[dtype_status, dtype_preview]
            )
            dtype_undo_btn.click(fn=undo_last_action, outputs=[dtype_status, dtype_preview])
        
        # 5. Standardize Text
        with gr.Accordion("5️⃣ Standardize Text", open=False):
            with gr.Row():
                text_cols = gr.Dropdown(
                    choices=[],
                    label="Select Text Columns",
                    multiselect=True,
                    interactive=True
                )
                text_operation = gr.Dropdown(
                    choices=["Lowercase", "Uppercase", "Title Case", 
                            "Strip whitespace", "Remove special characters"],
                    label="Operation",
                    value="Lowercase"
                )
            
            with gr.Row():
                text_btn = gr.Button("Apply", variant="primary")
                text_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            text_status = gr.Textbox(label="Status", interactive=False)
            text_preview = gr.Dataframe(label="Preview", height=300)
            
            text_btn.click(
                fn=standardize_text,
                inputs=[text_cols, text_operation],
                outputs=[text_status, text_preview]
            )
            text_undo_btn.click(fn=undo_last_action, outputs=[text_status, text_preview])
        
        # 6. Scaling & Normalization
        with gr.Accordion("6️⃣ Scaling & Normalization", open=False):
            with gr.Row():
                scale_cols = gr.Dropdown(
                    choices=[],
                    label="Select Numeric Columns",
                    multiselect=True,
                    interactive=True
                )
                scale_method = gr.Dropdown(
                    choices=["Standard Scaler (Z-score)", "Min-Max Scaler (0-1)", "Robust Scaler"],
                    label="Scaling Method",
                    value="Standard Scaler (Z-score)",
                    info="Standard: mean=0, std=1 | MinMax: range 0-1 | Robust: uses median"
                )
            
            with gr.Row():
                scale_btn = gr.Button("Apply", variant="primary")
                scale_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            scale_status = gr.Textbox(label="Status", interactive=False)
            scale_preview = gr.Dataframe(label="Preview", height=300)
            
            scale_btn.click(
                fn=scale_normalize,
                inputs=[scale_cols, scale_method],
                outputs=[scale_status, scale_preview]
            )
            scale_undo_btn.click(fn=undo_last_action, outputs=[scale_status, scale_preview])
        
        gr.Markdown("---")
        
        # Download section
        with gr.Row():
            download_btn = gr.Button("📥 Download Cleaned Data", variant="secondary", size="lg")
        
        download_file = gr.File(label="Download CSV")
        
        download_btn.click(fn=download_cleaned_data, outputs=download_file)

# ========================================================================
    # TAB 3: EDA (EXPLORATORY DATA ANALYSIS)
    # ========================================================================
    
    with gr.Tab("📊 EDA (Exploratory Data Analysis)"):
        gr.Markdown("### Comprehensive Exploratory Data Analysis Tools")
        
        # 1. Understanding Data
        with gr.Accordion("1️⃣ Data Overview", open=True):
            understand_btn = gr.Button("📋 Show Complete Data Overview", variant="primary")
            understand_output = gr.Markdown()
            understand_table = gr.Dataframe(label="Sample Data (First 10 rows)", height=300)
            
            understand_btn.click(
                fn=understand_data,
                outputs=[understand_output, understand_table]
            )
        
        # 2. Descriptive Statistics
        with gr.Accordion("2️⃣ Descriptive Statistics", open=False):
            with gr.Row():
                desc_type = gr.Radio(
                    choices=["Numeric Only", "All Columns", "Categorical Only"],
                    label="Statistics Type",
                    value="Numeric Only",
                    info="Choose which types of columns to analyze"
                )
            desc_btn = gr.Button("📊 Generate Statistics", variant="primary")
            desc_output = gr.Dataframe(label="Descriptive Statistics", height=400)
            
            desc_btn.click(fn=descriptive_stats, inputs=desc_type, outputs=desc_output)
        
        # 3. Univariate Analysis
        with gr.Accordion("3️⃣ Univariate Analysis (Single Variable)", open=False):
            with gr.Row():
                uni_col = gr.Dropdown(choices=[], label="Select Column", interactive=True)
                uni_plot_type = gr.Dropdown(
                    choices=["Histogram", "Box Plot", "Value Counts", "Statistics"],
                    label="Analysis Type",
                    value="Histogram",
                    info="Histogram: distribution | Box Plot: outliers | Value Counts: frequency"
                )
            uni_btn = gr.Button("🔍 Analyze", variant="primary")
            uni_output = gr.Plot(label="Visualization")
            uni_stats = gr.Markdown(label="Statistics")
            
            uni_btn.click(
                fn=univariate_analysis,
                inputs=[uni_col, uni_plot_type],
                outputs=[uni_output, uni_stats]
            )
        
        # 4. Bivariate Analysis
        with gr.Accordion("4️⃣ Bivariate Analysis (Two Variables)", open=False):
            with gr.Row():
                bi_col1 = gr.Dropdown(choices=[], label="Select Column 1", interactive=True)
                bi_col2 = gr.Dropdown(choices=[], label="Select Column 2", interactive=True)
                bi_plot_type = gr.Dropdown(
                    choices=["Scatter Plot", "Line Plot", "Bar Plot", "Correlation"],
                    label="Plot Type",
                    value="Scatter Plot",
                    info="Scatter: relationship | Line: trend | Bar: comparison | Correlation: strength"
                )
            bi_btn = gr.Button("🔍 Analyze", variant="primary")
            bi_output = gr.Plot(label="Visualization")
            bi_stats = gr.Markdown(label="Statistics")
            
            bi_btn.click(
                fn=bivariate_analysis,
                inputs=[bi_col1, bi_col2, bi_plot_type],
                outputs=[bi_output, bi_stats]
            )
        
        # 5. Correlation Matrix
        with gr.Accordion("5️⃣ Correlation Matrix (Multivariate)", open=False):
            with gr.Row():
                corr_method = gr.Dropdown(
                    choices=["pearson", "spearman", "kendall"],
                    label="Correlation Method",
                    value="pearson",
                    info="Pearson: linear | Spearman: monotonic | Kendall: ordinal"
                )
            corr_btn = gr.Button("🔥 Generate Correlation Heatmap", variant="primary")
            corr_output = gr.Plot(label="Correlation Heatmap", height=600)
            
            corr_btn.click(fn=correlation_matrix, inputs=corr_method, outputs=corr_output)
        
        # 6. Missing Value Analysis
        with gr.Accordion("6️⃣ Missing Value Analysis", open=False):
            missing_analysis_btn = gr.Button("🔍 Analyze Missing Values", variant="primary")
            missing_plot = gr.Plot(label="Missing Values Visualization")
            missing_details = gr.Markdown(label="Detailed Analysis")
            
            missing_analysis_btn.click(
                fn=missing_value_analysis,
                outputs=[missing_plot, missing_details]
            )
        
        # 7. Outlier Detection
        with gr.Accordion("7️⃣ Outlier Detection", open=False):
            outlier_col_eda = gr.Dropdown(choices=[], label="Select Numeric Column", interactive=True)
            outlier_method_vis = gr.Dropdown(
                choices=["Box Plot", "IQR Analysis", "Z-Score Analysis"],
                label="Detection Method",
                value="Box Plot",
                info="Box Plot: visual | IQR: statistical | Z-Score: deviation"
            )
            outlier_btn_eda = gr.Button("🔍 Detect Outliers", variant="primary")
            outlier_plot_eda = gr.Plot(label="Visualization")
            outlier_stats_eda = gr.Markdown(label="Outlier Statistics")
            
            outlier_btn_eda.click(
                fn=outlier_detection,
                inputs=[outlier_col_eda, outlier_method_vis],
                outputs=[outlier_plot_eda, outlier_stats_eda]
            )
        
        # 8. Distribution Analysis
        with gr.Accordion("8️⃣ Distribution Analysis", open=False):
            dist_col = gr.Dropdown(choices=[], label="Select Numeric Column", interactive=True)
            dist_btn = gr.Button("📈 Analyze Distribution", variant="primary")
            dist_plot = gr.Plot(label="Distribution Plots (Histogram + Q-Q Plot)")
            dist_stats = gr.Markdown(label="Distribution Statistics")
            
            dist_btn.click(
                fn=distribution_analysis,
                inputs=dist_col,
                outputs=[dist_plot, dist_stats]
            )
        
        # 9. Categorical Data Analysis
        with gr.Accordion("9️⃣ Categorical Data Analysis", open=False):
            cat_col = gr.Dropdown(choices=[], label="Select Categorical Column", interactive=True)
            cat_btn = gr.Button("📊 Analyze Categories", variant="primary")
            cat_plot = gr.Plot(label="Category Visualization (Bar + Pie)")
            cat_stats = gr.Markdown(label="Category Statistics")
            
            cat_btn.click(
                fn=categorical_analysis,
                inputs=cat_col,
                outputs=[cat_plot, cat_stats]
            )

# ========================================================================
    # TAB 4: FEATURE ENGINEERING
    # ========================================================================
    
    with gr.Tab("⚙️ Feature Engineering"):
        gr.Markdown("### Feature Engineering & Transformation Tools")
        
        # 1. Feature Creation
        with gr.Accordion("1️⃣ Feature Creation", open=False):
            gr.Markdown("Create new features by combining or transforming existing ones")
            
            with gr.Row():
                fc_operation = gr.Dropdown(
                    choices=["Combine (Add)", "Combine (Multiply)", "Combine (Divide)", 
                            "Mathematical Transform", "DateTime Features"],
                    label="Operation Type",
                    value="Combine (Add)",
                    info="Choose how to create the new feature"
                )
            
            with gr.Row():
                fc_col1 = gr.Dropdown(choices=[], label="Select Column 1", interactive=True)
                fc_col2 = gr.Dropdown(choices=[], label="Select Column 2 (if combining)", interactive=True)
                fc_new_name = gr.Textbox(label="New Feature Name", placeholder="new_feature")
            
            with gr.Row():
                fc_math_op = gr.Dropdown(
                    choices=["Square", "Cube", "Square Root", "Absolute"],
                    label="Math Operation (if Mathematical Transform)",
                    value="Square"
                )
            
            with gr.Row():
                fc_btn = gr.Button("✨ Create Feature", variant="primary")
                fc_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            fc_status = gr.Textbox(label="Status", interactive=False)
            fc_preview = gr.Dataframe(label="Preview", height=300)
            
            fc_btn.click(
                fn=create_feature,
                inputs=[fc_operation, fc_col1, fc_col2, fc_new_name, fc_math_op],
                outputs=[fc_status, fc_preview]
            )
            fc_undo_btn.click(fn=undo_last_action, outputs=[fc_status, fc_preview])
        
        # 2. Feature Transformation
        with gr.Accordion("2️⃣ Feature Transformation", open=False):
            with gr.Row():
                ft_cols = gr.Dropdown(
                    choices=[], 
                    label="Select Columns", 
                    multiselect=True, 
                    interactive=True
                )
                ft_method = gr.Dropdown(
                    choices=["Log Transform", "Square Transform", "Power Transform (Yeo-Johnson)"],
                    label="Transformation Method",
                    value="Log Transform",
                    info="Log: for right-skewed | Square: amplify values | Power: normalize"
                )
            
            with gr.Row():
                ft_btn = gr.Button("🔄 Transform", variant="primary")
                ft_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            ft_status = gr.Textbox(label="Status", interactive=False)
            ft_preview = gr.Dataframe(label="Preview", height=300)
            
            ft_btn.click(
                fn=transform_features,
                inputs=[ft_cols, ft_method],
                outputs=[ft_status, ft_preview]
            )
            ft_undo_btn.click(fn=undo_last_action, outputs=[ft_status, ft_preview])
        
        # 3. Encoding Categorical Variables
        with gr.Accordion("3️⃣ Encoding Categorical Variables", open=False):
            with gr.Row():
                enc_cols = gr.Dropdown(
                    choices=[], 
                    label="Select Columns", 
                    multiselect=True, 
                    interactive=True
                )
                enc_method = gr.Dropdown(
                    choices=["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"],
                    label="Encoding Method",
                    value="Label Encoding",
                    info="Label: 0,1,2... | One-Hot: binary columns | Ordinal: custom order"
                )
            
            with gr.Row():
                enc_btn = gr.Button("🔢 Encode", variant="primary")
                enc_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            enc_status = gr.Textbox(label="Status", interactive=False)
            enc_preview = gr.Dataframe(label="Preview", height=300)
            
            enc_btn.click(
                fn=encode_features,
                inputs=[enc_cols, enc_method],
                outputs=[enc_status, enc_preview]
            )
            enc_undo_btn.click(fn=undo_last_action, outputs=[enc_status, enc_preview])
        
        # 4. Binning/Discretization
        with gr.Accordion("4️⃣ Binning / Discretization", open=False):
            with gr.Row():
                bin_col = gr.Dropdown(choices=[], label="Select Column", interactive=True)
                bin_method = gr.Dropdown(
                    choices=["Equal Width", "Equal Frequency"],
                    label="Binning Method",
                    value="Equal Width",
                    info="Equal Width: same range | Equal Frequency: same count"
                )
                bin_count = gr.Number(label="Number of Bins", value=5, minimum=2, maximum=20)
            
            with gr.Row():
                bin_btn = gr.Button("📦 Apply Binning", variant="primary")
                bin_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            bin_status = gr.Textbox(label="Status", interactive=False)
            bin_preview = gr.Dataframe(label="Preview", height=300)
            
            bin_btn.click(
                fn=apply_binning,
                inputs=[bin_col, bin_method, bin_count],
                outputs=[bin_status, bin_preview]
            )
            bin_undo_btn.click(fn=undo_last_action, outputs=[bin_status, bin_preview])
        
        # 5. Dimensionality Reduction
        with gr.Accordion("5️⃣ Dimensionality Reduction (PCA)", open=False):
            with gr.Row():
                dr_method = gr.Dropdown(
                    choices=["PCA (Principal Component Analysis)"],
                    label="Method",
                    value="PCA (Principal Component Analysis)"
                )
                dr_components = gr.Number(
                    label="Number of Components", 
                    value=2, 
                    minimum=1,
                    info="Number of principal components to create"
                )
            
            with gr.Row():
                dr_btn = gr.Button("📉 Apply Reduction", variant="primary")
                dr_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            dr_status = gr.Textbox(label="Status", interactive=False)
            dr_preview = gr.Dataframe(label="Preview", height=300)
            dr_plot = gr.Plot(label="Explained Variance")
            
            dr_btn.click(
                fn=reduce_dimensions,
                inputs=[dr_method, dr_components],
                outputs=[dr_status, dr_preview, dr_plot]
            )
            dr_undo_btn.click(fn=undo_last_action, outputs=[dr_status, dr_preview])
        
        # 6. Polynomial Features
        with gr.Accordion("6️⃣ Polynomial Features (Interaction)", open=False):
            with gr.Row():
                poly_cols = gr.Dropdown(
                    choices=[], 
                    label="Select Columns", 
                    multiselect=True, 
                    interactive=True,
                    info="Select features to create polynomial interactions"
                )
                poly_degree = gr.Number(
                    label="Polynomial Degree", 
                    value=2, 
                    minimum=2, 
                    maximum=3,
                    info="Degree 2: x², xy | Degree 3: x³, x²y, xy², y³"
                )
            
            with gr.Row():
                poly_btn = gr.Button("🔗 Create Polynomial Features", variant="primary")
                poly_undo_btn = gr.Button("↩️ Undo", variant="secondary")
            
            poly_status = gr.Textbox(label="Status", interactive=False)
            poly_preview = gr.Dataframe(label="Preview", height=300)
            
            poly_btn.click(
                fn=create_polynomial,
                inputs=[poly_cols, poly_degree],
                outputs=[poly_status, poly_preview]
            )
            poly_undo_btn.click(fn=undo_last_action, outputs=[poly_status, poly_preview])

# ========================================================================
    # TAB 5: MODEL BUILDING
    # ========================================================================
    
    with gr.Tab("🤖 Model Building"):
        gr.Markdown("### Complete Machine Learning Pipeline")
        
        # 1. Problem Formulation
        with gr.Accordion("1️⃣ Problem Formulation", open=True):
            gr.Markdown("#### Define your machine learning task")
            
            with gr.Row():
                target_info_btn = gr.Button("📋 Show Target Variable Guide", variant="secondary")
            
            target_info_output = gr.Markdown()
            
            with gr.Row():
                mb_target_col = gr.Dropdown(
                    choices=[], 
                    label="Select Target Variable (Y)", 
                    interactive=True,
                    info="The variable you want to predict"
                )
                mb_task_type = gr.Radio(
                    choices=["Classification", "Regression", "Clustering"],
                    label="Task Type",
                    value="Classification",
                    info="Classification: categories | Regression: numbers | Clustering: patterns"
                )
            
            target_info_btn.click(fn=get_target_info, outputs=target_info_output)
        
        # 2. Data Splitting
        with gr.Accordion("2️⃣ Data Splitting", open=False):
            with gr.Row():
                split_test_size = gr.Slider(
                    0.1, 0.5, value=0.2, step=0.05, 
                    label="Test Set Size",
                    info="Proportion of data for testing (typically 0.2)"
                )
                split_val_size = gr.Slider(
                    0.1, 0.3, value=0.15, step=0.05, 
                    label="Validation Set Size",
                    info="Used only if validation is enabled"
                )
                use_validation = gr.Checkbox(
                    label="Use Validation Set", 
                    value=False,
                    info="Creates a third dataset for model tuning"
                )
            
            with gr.Row():
                split_btn = gr.Button("✂️ Split Data", variant="primary", size="lg")
            
            split_output = gr.Markdown()
            split_train_preview = gr.Dataframe(label="Training Set Preview", visible=False, height=200)
            split_test_preview = gr.Dataframe(label="Test Set Preview", visible=False, height=200)
            
            split_btn.click(
                fn=prepare_data_split,
                inputs=[mb_target_col, mb_task_type, split_test_size, split_val_size, use_validation],
                outputs=[split_output, split_train_preview, split_test_preview, gr.Dataframe(visible=False), gr.Dataframe(visible=False)]
            )
        
        # 3. Feature Selection
        with gr.Accordion("3️⃣ Feature Selection", open=False):
            gr.Markdown("#### Select the most important features for modeling")
            
            with gr.Row():
                fs_method = gr.Dropdown(
                    choices=["Correlation with Target", "F-Test (ANOVA)", "Recursive Feature Elimination (RFE)"],
                    label="Selection Method",
                    value="F-Test (ANOVA)",
                    info="Correlation: simple | F-Test: statistical | RFE: model-based"
                )
                fs_n_features = gr.Number(
                    label="Number of Features to Select", 
                    value=10,
                    minimum=1,
                    info="Top N most important features"
                )
            
            with gr.Row():
                fs_btn = gr.Button("🎯 Select Features", variant="primary")
            
            fs_output = gr.Markdown()
            
            fs_btn.click(
                fn=perform_feature_selection,
                inputs=[fs_method, fs_n_features],
                outputs=[fs_output, gr.Textbox(visible=False)]
            )
        
        # 4. Model Training
        with gr.Accordion("4️⃣ Model Training", open=False):
            gr.Markdown("#### Choose and train a machine learning algorithm")
            
            with gr.Row():
                algorithm_choice = gr.Dropdown(
                    choices=[
                        "Linear Regression",
                        "Ridge Regression",
                        "Lasso Regression",
                        "Logistic Regression",
                        "Decision Tree (Classification)",
                        "Decision Tree (Regression)",
                        "Random Forest (Classification)",
                        "Random Forest (Regression)",
                        "Gradient Boosting (Classification)",
                        "Gradient Boosting (Regression)",
                        "K-Nearest Neighbors (Classification)",
                        "K-Nearest Neighbors (Regression)",
                        "Naive Bayes",
                        "Support Vector Machine (Classification)",
                        "Support Vector Machine (Regression)"
                    ],
                    label="Select Algorithm",
                    value="Random Forest (Classification)",
                    info="Choose based on your task type"
                )
            
            with gr.Row():
                train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
            
            train_output = gr.Markdown()
            train_plot = gr.Plot(label="Training Visualization", height=400)
            train_status = gr.Textbox(label="Training Status", interactive=False)
            
            train_btn.click(
                fn=train_model,
                inputs=algorithm_choice,
                outputs=[train_output, train_plot, train_status]
            )
        
        # 5. Model Evaluation
        with gr.Accordion("5️⃣ Model Evaluation", open=False):
            gr.Markdown("#### Evaluate model performance on test data")
            
            with gr.Row():
                eval_btn = gr.Button("📊 Evaluate Model", variant="primary", size="lg")
            
            eval_output = gr.Markdown()
            eval_plot = gr.Plot(label="Feature Importance / Metrics")
            
            eval_btn.click(fn=evaluate_model, outputs=[eval_output, eval_plot])
        
        # 6. Cross-Validation
        with gr.Accordion("6️⃣ Cross-Validation", open=False):
            gr.Markdown("#### Assess model stability with k-fold cross-validation")
            
            with gr.Row():
                cv_k_folds = gr.Number(
                    label="Number of Folds (K)", 
                    value=5,
                    minimum=2,
                    maximum=10,
                    info="Typically 5 or 10 folds"
                )
            
            with gr.Row():
                cv_btn = gr.Button("🔄 Perform Cross-Validation", variant="primary")
            
            cv_output = gr.Markdown()
            
            cv_btn.click(fn=perform_cross_validation, inputs=cv_k_folds, outputs=cv_output)
        
        # 7. Hyperparameter Tuning
        with gr.Accordion("7️⃣ Hyperparameter Tuning", open=False):
            gr.Markdown("#### Optimize model parameters for better performance")
            
            with gr.Row():
                hp_algorithm = gr.Dropdown(
                    choices=[
                        "Random Forest (Classification)",
                        "Random Forest (Regression)",
                        "Gradient Boosting (Classification)",
                        "Gradient Boosting (Regression)",
                        "Logistic Regression",
                        "K-Nearest Neighbors (Classification)",
                        "K-Nearest Neighbors (Regression)"
                    ],
                    label="Algorithm",
                    value="Random Forest (Classification)",
                    info="Select algorithm to tune"
                )
                hp_search_method = gr.Radio(
                    choices=["Grid Search", "Random Search"],
                    label="Search Method",
                    value="Grid Search",
                    info="Grid: exhaustive | Random: faster"
                )
                hp_n_iter = gr.Number(
                    label="Iterations (Random Search)", 
                    value=10,
                    minimum=5,
                    maximum=50,
                    info="Number of random combinations to try"
                )
            
            with gr.Row():
                hp_btn = gr.Button("⚙️ Tune Hyperparameters", variant="primary")
            
            hp_output = gr.Markdown()
            hp_status = gr.Textbox(label="Tuning Status", interactive=False)
            
            hp_btn.click(
                fn=hyperparameter_tuning,
                inputs=[hp_algorithm, hp_search_method, hp_n_iter],
                outputs=[hp_output, hp_status]
            )
        
        # 8. Handle Class Imbalance
        with gr.Accordion("8️⃣ Handle Class Imbalance (Classification)", open=False):
            gr.Markdown("#### Address imbalanced datasets")
            
            with gr.Row():
                imb_method = gr.Radio(
                    choices=["SMOTE (Oversampling)", "Class Weights"],
                    label="Method",
                    value="SMOTE (Oversampling)",
                    info="SMOTE: creates synthetic samples | Weights: adjusts importance"
                )
            
            with gr.Row():
                imb_btn = gr.Button("⚖️ Apply Method", variant="primary")
            
            imb_output = gr.Markdown()
            
            imb_btn.click(
                fn=handle_imbalanced_data,
                inputs=imb_method,
                outputs=[imb_output, gr.Dataframe(visible=False), gr.Dataframe(visible=False)]
            )
        
        # 9. Clustering Analysis
        with gr.Accordion("9️⃣ Clustering Analysis (Unsupervised)", open=False):
            gr.Markdown("#### Discover patterns without labels")
            
            with gr.Row():
                cluster_algorithm = gr.Dropdown(
                    choices=["K-Means", "DBSCAN", "Hierarchical"],
                    label="Algorithm",
                    value="K-Means",
                    info="K-Means: centroid-based | DBSCAN: density-based | Hierarchical: tree-based"
                )
                cluster_n = gr.Number(
                    label="Number of Clusters", 
                    value=3,
                    minimum=2,
                    maximum=10,
                    info="For K-Means and Hierarchical only"
                )
            
            with gr.Row():
                cluster_btn = gr.Button("🎯 Perform Clustering", variant="primary")
            
            cluster_output = gr.Markdown()
            cluster_plot = gr.Plot(label="Cluster Visualization")
            cluster_status = gr.Textbox(label="Status", interactive=False)
            
            cluster_btn.click(
                fn=perform_clustering,
                inputs=[cluster_algorithm, cluster_n],
                outputs=[cluster_output, cluster_plot, cluster_status]
            )
        
        # 10. Model Deployment
        with gr.Accordion("🔟 Model Deployment", open=False):
            gr.Markdown("#### Save your trained model for future use")
            
            with gr.Row():
                save_filename = gr.Textbox(
                    label="Model Filename", 
                    placeholder="my_model", 
                    value="trained_model",
                    info="Model will be saved as .pkl file"
                )
            
            with gr.Row():
                save_btn = gr.Button("💾 Save Model", variant="primary", size="lg")
            
            save_status = gr.Textbox(label="Save Status", interactive=False, lines=3)
            
            save_btn.click(fn=save_trained_model, inputs=save_filename, outputs=save_status)

# ========================================================================
    # EVENT HANDLERS - Connect file upload to all dropdowns
    # ========================================================================
    
    # Update ALL dropdowns when file is uploaded
    file_input.change(
        fn=load_file_extended,
        inputs=file_input,
        outputs=[
            # Basic outputs
            info_output, 
            table_output,
            # Data Cleaning dropdowns
            missing_cols,      # 2
            dup_cols,          # 3
            outlier_cols,      # 4
            dtype_col,         # 5
            text_cols,         # 6
            scale_cols,        # 7
            # EDA dropdowns
            uni_col,           # 8
            bi_col1,           # 9
            bi_col2,           # 10
            outlier_col_eda,   # 11
            dist_col,          # 12
            cat_col,           # 13
            # Feature Engineering dropdowns (FIXED!)
            fc_col1,           # 14
            fc_col2,           # 15
            ft_cols,           # 16
            enc_cols,          # 17
            bin_col,           # 18
            poly_cols,         # 19
            # Model Building dropdown
            mb_target_col      # 20
        ]
    )

# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Advanced EDA & ML Application")
    print("=" * 70)
    print("\n📊 Features:")
    print("  ✓ Data Cleaning & Preprocessing")
    print("  ✓ Exploratory Data Analysis")
    print("  ✓ Feature Engineering")
    print("  ✓ Machine Learning Pipeline")
    print("  ✓ Model Evaluation & Tuning")
    print("\n🌐 Access the app at: http://localhost:7860")
    print("=" * 70)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
