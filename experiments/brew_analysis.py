#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoost, Pool
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# File path
BREW_DATA_PATH = '../data/nlc-brew-data.csv'

def load_brew_data(sample_size=None):
    """Load the brew data from CSV file with optional sampling"""
    print("Loading brew data...")
    
    try:
        # If sample_size is provided, read only that many rows
        if sample_size:
            brew_data = pd.read_csv(BREW_DATA_PATH, nrows=sample_size)
        else:
            # Try to read the entire file, but use chunking for very large files
            try:
                brew_data = pd.read_csv(BREW_DATA_PATH)
            except:
                print("File too large, reading in chunks and sampling...")
                # Read in chunks and sample
                chunks = []
                for chunk in pd.read_csv(BREW_DATA_PATH, chunksize=100000):
                    chunks.append(chunk.sample(min(10000, len(chunk))))
                brew_data = pd.concat(chunks)
                
        print(f"Loaded brew data: {brew_data.shape[0]} rows, {brew_data.shape[1]} columns")
        return brew_data
    except Exception as e:
        print(f"Error loading brew data: {e}")
        return None

def explore_brew_data(brew_data):
    """Perform exploratory data analysis on the brew dataset"""
    if brew_data is None:
        return
    
    print("\n===== Exploratory Data Analysis for Brew Data =====")
    
    # Basic information
    print("\nBasic Information:")
    print(f"Number of rows: {brew_data.shape[0]}")
    print(f"Number of columns: {brew_data.shape[1]}")
    
    # Column information
    print("\nColumns and data types:")
    print(brew_data.dtypes)
    
    # Summary statistics for numerical columns
    print("\nSummary statistics:")
    print(brew_data.describe())
    
    # Missing values
    print("\nMissing values per column:")
    missing_values = brew_data.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Sample data
    print("\nSample data (first 5 rows):")
    print(brew_data.head())
    
    # Save detailed column information to a file
    with open('brew_data_details.txt', 'w') as f:
        f.write("Brew Data Column Details:\n\n")
        
        for col in brew_data.columns:
            f.write(f"Column: {col}\n")
            f.write(f"Data type: {brew_data[col].dtype}\n")
            
            if brew_data[col].dtype == 'object':
                # For string columns, show unique values if not too many
                unique_values = brew_data[col].nunique()
                if unique_values < 20:
                    f.write(f"Unique values ({unique_values}): {brew_data[col].unique()}\n")
                else:
                    f.write(f"Unique values: {unique_values} (too many to display)\n")
                    
                # Show most common values
                f.write("Most common values:\n")
                for val, count in brew_data[col].value_counts().head(5).items():
                    f.write(f"  - {val}: {count} occurrences\n")
            else:
                # For numeric columns, show basic statistics
                f.write(f"Min: {brew_data[col].min()}\n")
                f.write(f"Max: {brew_data[col].max()}\n")
                f.write(f"Mean: {brew_data[col].mean()}\n")
                f.write(f"Median: {brew_data[col].median()}\n")
            
            f.write("\n")
    
    print("\nDetailed column information saved to 'brew_data_details.txt'")

def correlation_analysis(brew_data):
    """Perform correlation analysis on brew data"""
    if brew_data is None:
        return None
    
    print("\n===== Correlation Analysis for Brew Data =====")
    
    # Create output directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Convert non-numeric columns to categorical codes for correlation
    numerical_data = brew_data.copy()
    for col in numerical_data.columns:
        if numerical_data[col].dtype == 'object':
            try:
                numerical_data[col] = numerical_data[col].astype('category').cat.codes
            except:
                print(f"Could not convert column {col} to categorical")
                numerical_data = numerical_data.drop(columns=[col])
    
    # Handle large datasets by sampling
    if numerical_data.shape[0] > 10000:
        numerical_data = numerical_data.sample(10000, random_state=42)
    
    # Calculate correlation matrix
    try:
        correlation_matrix = numerical_data.corr()
        
        # Top positive correlations (excluding self-correlations)
        print("\nTop positive correlations:")
        pos_corr = correlation_matrix.unstack().sort_values(ascending=False)
        # Filter out self-correlations (which are always 1.0)
        pos_corr = pos_corr[pos_corr < 1.0]
        print(pos_corr.head(15))
        
        # Top negative correlations
        print("\nTop negative correlations:")
        neg_corr = correlation_matrix.unstack().sort_values(ascending=True)
        print(neg_corr.head(15))
        
        # Plot correlation matrix (using a mask to show only half)
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=False)
        
        plt.title('Correlation Matrix - Brew Data', fontsize=16)
        plt.tight_layout()
        plt.savefig('figures/brew_correlation_matrix.png', dpi=300)
        plt.close()
        
        # Save correlation values to file
        correlation_matrix.to_csv('brew_correlation_matrix.csv')
        print("Correlation matrix saved to 'brew_correlation_matrix.csv'")
        
        # Plot top correlations for key target variables
        potential_targets = ['coffeeWeight', 'brewTime', 'TDS', 'extractionYield']
        
        # Check which target variables exist in the data
        existing_targets = [col for col in potential_targets if col in correlation_matrix.columns]
        
        if existing_targets:
            for target in existing_targets:
                # Get top 10 correlations for this target
                target_corr = correlation_matrix[target].sort_values(ascending=False)
                target_corr = target_corr[target_corr.index != target]  # Remove self
                
                # Plot top 10 correlations
                plt.figure(figsize=(10, 8))
                sns.barplot(x=target_corr.head(10), y=target_corr.head(10).index)
                plt.title(f'Top Correlations with {target}', fontsize=14)
                plt.xlabel('Correlation Coefficient', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'figures/brew_{target}_correlations.png', dpi=300)
                plt.close()
                
                print(f"\nTop correlations with {target}:")
                print(target_corr.head(10))
        
        return correlation_matrix
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        return None

def preprocess_brew_data(brew_data, target_variable='extractionYield'):
    """Preprocess brew data for machine learning models"""
    if brew_data is None:
        return None, None
    
    print(f"\n===== Preprocessing Brew Data for {target_variable} Prediction =====")
    
    # Create a copy to avoid modifying the original data
    df = brew_data.copy()
    
    # Convert datetime columns if they exist
    datetime_cols = ['eventTime', 'brewTime', 'startTime', 'endTime']
    for col in datetime_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                print(f"Warning: Could not convert {col} to datetime")
    
    # Handle list-like columns stored as strings
    list_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                df[col].dropna().iloc[0].startswith('[') if not df[col].empty]
    
    for col in list_cols:
        try:
            # Convert string representations to actual lists
            df[col] = df[col].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x])
            
            # Create features based on list statistics
            df[f'{col}_len'] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 1)
            df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else x)
            df[f'{col}_max'] = df[col].apply(lambda x: max(x) if isinstance(x, list) and len(x) > 0 else x)
            df[f'{col}_min'] = df[col].apply(lambda x: min(x) if isinstance(x, list) and len(x) > 0 else x)
        except Exception as e:
            print(f"Warning: Error processing list column {col}: {e}")
    
    # Check if target variable exists
    if target_variable not in df.columns:
        print(f"Target variable '{target_variable}' not found in data!")
        # Try to find an alternative
        alternative_targets = ['coffeeWeight', 'brewTime', 'TDS', 'extractionYield', 'weight', 'brewRatio']
        for alt in alternative_targets:
            if alt in df.columns and alt != target_variable:
                target_variable = alt
                print(f"Using alternative target: {target_variable}")
                break
        else:
            # If no alternative is found
            print("No suitable target variable found.")
            return None, None
    
    # Select numerical features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove the target from features if it's in the list
    if target_variable in numeric_cols:
        numeric_cols.remove(target_variable)
    
    print(f"Selected {len(numeric_cols)} numerical features")
    
    # Drop rows with missing values in the target or features
    df_clean = df[numeric_cols + [target_variable]].dropna()
    print(f"Data shape after removing rows with missing values: {df_clean.shape}")
    
    if df_clean.empty:
        print("Error: No data left after preprocessing!")
        return None, None
    
    # Extract features and target
    X = df_clean[numeric_cols]
    y = df_clean[target_variable]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def run_models(X, y, target_name):
    """Train and evaluate XGBoost and CatBoost models"""
    if X is None or y is None:
        return None, None
    
    # Create output directory for figures
    os.makedirs('figures', exist_ok=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost model
    print("\n===== XGBoost Model =====")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # XGBoost predictions
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    print(f"XGBoost RMSE: {xgb_rmse:.4f}")
    print(f"XGBoost R²: {xgb_r2:.4f}")
    
    # Feature importance for XGBoost
    xgb_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nXGBoost Feature Importance (top 10):")
    print(xgb_importance.head(10))
    
    # Plot XGBoost feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=xgb_importance.head(15))
    plt.title(f'XGBoost Feature Importance for {target_name} Prediction')
    plt.tight_layout()
    plt.savefig(f'figures/xgb_importance_{target_name}.png')
    plt.close()
    
    # CatBoost model
    print("\n===== CatBoost Model =====")
    cat_model = CatBoost({
        'iterations': 100,
        'depth': 3,
        'learning_rate': 0.1,
        'loss_function': 'RMSE',
        'verbose': 50
    })
    
    # Create pool objects
    train_pool = Pool(X_train_scaled, y_train)
    test_pool = Pool(X_test_scaled, y_test)
    
    cat_model.fit(train_pool)
    
    # CatBoost predictions
    cat_pred = cat_model.predict(test_pool)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
    cat_r2 = r2_score(y_test, cat_pred)
    
    print(f"CatBoost RMSE: {cat_rmse:.4f}")
    print(f"CatBoost R²: {cat_r2:.4f}")
    
    # Feature importance for CatBoost
    cat_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': cat_model.get_feature_importance()
    }).sort_values(by='Importance', ascending=False)
    
    print("\nCatBoost Feature Importance (top 10):")
    print(cat_importance.head(10))
    
    # Plot CatBoost feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=cat_importance.head(15))
    plt.title(f'CatBoost Feature Importance for {target_name} Prediction')
    plt.tight_layout()
    plt.savefig(f'figures/cat_importance_{target_name}.png')
    plt.close()
    
    # Compare predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, xgb_pred, alpha=0.5, label=f'XGBoost (R²={xgb_r2:.3f})')
    plt.scatter(y_test, cat_pred, alpha=0.5, label=f'CatBoost (R²={cat_r2:.3f})')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'Actual vs Predicted {target_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/prediction_comparison_{target_name}.png')
    plt.close()
    
    # Model comparison summary
    print("\n===== Model Comparison =====")
    print(f"XGBoost RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")
    print(f"CatBoost RMSE: {cat_rmse:.4f}, R²: {cat_r2:.4f}")
    
    if xgb_rmse < cat_rmse:
        print("XGBoost performed better in terms of RMSE")
    else:
        print("CatBoost performed better in terms of RMSE")
    
    return xgb_model, cat_model

def main():
    """Main function to orchestrate the analysis"""
    # Load brew data (sample to handle large file)
    brew_data = load_brew_data(sample_size=50000)
    
    if brew_data is None:
        print("Failed to load brew data. Exiting.")
        return
    
    # Explore data
    explore_brew_data(brew_data)
    
    # Correlation analysis
    correlation_matrix = correlation_analysis(brew_data)
    
    # Try different target variables
    potential_targets = ['extractionYield', 'TDS', 'brewTime', 'coffeeWeight']
    
    for target in potential_targets:
        if target in brew_data.columns:
            print(f"\n\n===== Modeling for target: {target} =====")
            X, y = preprocess_brew_data(brew_data, target_variable=target)
            
            if X is not None and y is not None:
                xgb_model, cat_model = run_models(X, y, target)
            else:
                print(f"Skipping modeling for {target} due to preprocessing issues.")
    
    print("\nBrew data analysis complete!")

if __name__ == "__main__":
    main() 