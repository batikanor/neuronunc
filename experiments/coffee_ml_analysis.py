#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from catboost import CatBoost, Pool
import os
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

# File paths
GRIND_DATA_PATH = '../data/nlc-grind-data.csv'
BREW_DATA_PATH = '../data/nlc-brew-data.csv'

def load_data():
    """Load the coffee data from CSV files"""
    print("Loading coffee data...")
    
    # Load grind data
    try:
        grind_data = pd.read_csv(GRIND_DATA_PATH)
        print(f"Loaded grind data: {grind_data.shape[0]} rows, {grind_data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading grind data: {e}")
        grind_data = None
    
    # Load brew data - handle large file size
    try:
        # Read only first 100,000 rows if file is too large
        brew_data = pd.read_csv(BREW_DATA_PATH, nrows=100000)
        print(f"Loaded brew data: {brew_data.shape[0]} rows, {brew_data.shape[1]} columns")
    except Exception as e:
        print(f"Error loading brew data: {e}")
        brew_data = None
    
    return grind_data, brew_data

def exploratory_data_analysis(grind_data, brew_data):
    """Perform exploratory data analysis on the coffee datasets"""
    print("\n===== Exploratory Data Analysis =====")
    
    # Examine grind data
    if grind_data is not None:
        print("\n--- Grind Data Summary ---")
        print("Sample data:")
        print(grind_data.head())
        print("\nData types:")
        print(grind_data.dtypes)
        print("\nBasic statistics:")
        print(grind_data.describe())
        print("\nMissing values:")
        print(grind_data.isnull().sum())
        
        # Save column information to file
        with open('grind_data_columns.txt', 'w') as f:
            f.write("Grind Data Columns:\n")
            for col in grind_data.columns:
                f.write(f"- {col}: {grind_data[col].dtype}\n")
                f.write(f"  Sample values: {grind_data[col].head(3).tolist()}\n")
    
    # Examine brew data
    if brew_data is not None:
        print("\n--- Brew Data Summary ---")
        print("Sample data:")
        print(brew_data.head())
        print("\nData types:")
        print(brew_data.dtypes)
        print("\nBasic statistics:")
        print(brew_data.describe())
        print("\nMissing values:")
        print(brew_data.isnull().sum())
        
        # Save column information to file
        with open('brew_data_columns.txt', 'w') as f:
            f.write("Brew Data Columns:\n")
            for col in brew_data.columns:
                f.write(f"- {col}: {brew_data[col].dtype}\n")
                f.write(f"  Sample values: {brew_data[col].head(3).tolist()}\n")

def correlation_analysis(data, name, output_file):
    """Perform correlation analysis and save correlation matrix"""
    if data is None:
        return
    
    print(f"\n===== Correlation Analysis for {name} Data =====")
    
    # Convert string columns to categorical codes for correlation analysis
    numerical_data = data.copy()
    for col in numerical_data.columns:
        if numerical_data[col].dtype == 'object':
            numerical_data[col] = numerical_data[col].astype('category').cat.codes
    
    # Calculate correlation matrix
    try:
        # Handle large datasets by sampling if necessary
        if numerical_data.shape[0] > 10000:
            numerical_data = numerical_data.sample(10000, random_state=42)
        
        correlation_matrix = numerical_data.corr()
        
        # Top correlations
        print("\nTop positive correlations:")
        pos_corr = correlation_matrix.unstack().sort_values(ascending=False)
        print(pos_corr[pos_corr < 1.0].head(10))
        
        print("\nTop negative correlations:")
        neg_corr = correlation_matrix.unstack().sort_values(ascending=True)
        print(neg_corr.head(10))
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f'Correlation Matrix - {name} Data')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        print(f"Correlation matrix saved to {output_file}")
        
        return correlation_matrix
        
    except Exception as e:
        print(f"Error in correlation analysis: {e}")
        return None

def preprocess_grind_data(grind_data):
    """Preprocess grind data for modeling"""
    if grind_data is None:
        return None, None
    
    print("\n===== Preprocessing Grind Data =====")
    
    # Create a copy to avoid modifying the original data
    df = grind_data.copy()
    
    # Convert eventTime to datetime
    try:
        df['eventTime'] = pd.to_datetime(df['eventTime'])
    except:
        print("Warning: Could not convert eventTime to datetime")
    
    # Handle lists stored as strings (dosedWeights, shutterTimes)
    try:
        # Convert string representations of lists to actual lists
        if 'dosedWeights' in df.columns and df['dosedWeights'].dtype == 'object':
            df['dosedWeights'] = df['dosedWeights'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x])
            
            # Extract first dose weight as a feature
            df['first_dose_weight'] = df['dosedWeights'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            
            # Calculate number of doses
            df['num_doses'] = df['dosedWeights'].apply(lambda x: len(x) if isinstance(x, list) else 1)
        
        if 'shutterTimes' in df.columns and df['shutterTimes'].dtype == 'object':
            df['shutterTimes'] = df['shutterTimes'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else [x])
            
            # Extract first shutter time as a feature
            df['first_shutter_time'] = df['shutterTimes'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            
            # Calculate average shutter time
            df['avg_shutter_time'] = df['shutterTimes'].apply(
                lambda x: sum(x)/len(x) if isinstance(x, list) and len(x) > 0 else x)
    except Exception as e:
        print(f"Warning: Error processing list columns: {e}")
    
    # Handle categorical variables
    categorical_cols = ['eventType', 'shotType', 'sieveId']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    # Select features for modeling
    feature_cols = [
        'grindSize', 'recipeShutterTime', 'recipeBeanWeight', 
        'engineTemperature', 'totalDosedWeight'
    ]
    
    # Add derived features if they exist
    if 'first_dose_weight' in df.columns:
        feature_cols.append('first_dose_weight')
    if 'num_doses' in df.columns:
        feature_cols.append('num_doses')
    if 'first_shutter_time' in df.columns:
        feature_cols.append('first_shutter_time')
    if 'avg_shutter_time' in df.columns:
        feature_cols.append('avg_shutter_time')
    
    # Filter out features that don't exist in the data
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        print("Error: No valid features found in grind data")
        return None, None
    
    # Choose a target variable (e.g., totalDosedWeight or grindSize)
    target_col = 'totalDosedWeight'
    if target_col not in df.columns:
        target_col = 'grindSize'
        if target_col not in df.columns:
            print("Error: No valid target found in grind data")
            return None, None
    
    # Remove the target from features if it's there
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    # Drop rows with missing values in selected columns
    columns_to_use = feature_cols + [target_col]
    df_clean = df[columns_to_use].dropna()
    
    print(f"Features selected for modeling: {feature_cols}")
    print(f"Target variable: {target_col}")
    print(f"Processed data shape: {df_clean.shape}")
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    return X, y

def run_xgboost_model(X, y, dataset_name):
    """Train and evaluate XGBoost model"""
    if X is None or y is None:
        return None
    
    print(f"\n===== XGBoost Model for {dataset_name} =====")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Simple parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.01]
    }
    
    # Perform grid search (limited to save time)
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best XGBoost parameters: {grid_search.best_params_}")
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"XGBoost Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'XGBoost Feature Importance - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'xgboost_importance_{dataset_name}.png')
    plt.close()
    
    return best_model

def run_catboost_model(X, y, dataset_name):
    """Train and evaluate CatBoost model"""
    if X is None or y is None:
        return None
    
    print(f"\n===== CatBoost Model for {dataset_name} =====")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify categorical features
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'category']
    
    # Create pool objects
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    
    # Train CatBoost model
    model = CatBoost({
        'iterations': 200,
        'depth': 5,
        'learning_rate': 0.1,
        'loss_function': 'RMSE',
        'verbose': 100
    })
    
    model.fit(train_pool)
    
    # Predictions
    y_pred = model.predict(test_pool)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"CatBoost Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance()
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'CatBoost Feature Importance - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'catboost_importance_{dataset_name}.png')
    plt.close()
    
    return model

def compare_models(xgb_model, cat_model, X, y, dataset_name):
    """Compare XGBoost and CatBoost models"""
    if xgb_model is None or cat_model is None or X is None or y is None:
        return
    
    print(f"\n===== Model Comparison for {dataset_name} =====")
    
    # Cross-validation scores
    xgb_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    
    # For CatBoost we need to handle it differently due to categorical features
    cat_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'category']
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)
        
        model = CatBoost(cat_model.get_params())
        model.fit(train_pool, verbose=False)
        
        preds = model.predict(test_pool)
        score = -np.sqrt(mean_squared_error(y_test, preds))
        cat_scores.append(score)
    
    print("\nCross-validation RMSE scores:")
    print(f"  XGBoost: {-np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")
    print(f"  CatBoost: {-np.mean(cat_scores):.4f} ± {np.std(cat_scores):.4f}")
    
    # Compare feature importance
    xgb_importance = pd.DataFrame({
        'Feature': X.columns,
        'XGBoost_Importance': xgb_model.feature_importances_
    })
    
    cat_importance = pd.DataFrame({
        'Feature': X.columns,
        'CatBoost_Importance': cat_model.get_feature_importance()
    })
    
    importance_comparison = xgb_importance.merge(cat_importance, on='Feature')
    
    # Normalize importance scores for better comparison
    importance_comparison['XGBoost_Importance'] = importance_comparison['XGBoost_Importance'] / importance_comparison['XGBoost_Importance'].sum()
    importance_comparison['CatBoost_Importance'] = importance_comparison['CatBoost_Importance'] / importance_comparison['CatBoost_Importance'].sum()
    
    importance_comparison = importance_comparison.sort_values(by='XGBoost_Importance', ascending=False)
    
    print("\nFeature Importance Comparison:")
    print(importance_comparison)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Reshape data for plotting
    plot_data = pd.melt(
        importance_comparison, 
        id_vars=['Feature'], 
        value_vars=['XGBoost_Importance', 'CatBoost_Importance'],
        var_name='Model', 
        value_name='Importance'
    )
    
    sns.barplot(x='Feature', y='Importance', hue='Model', data=plot_data)
    plt.title(f'Feature Importance Comparison - {dataset_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'model_comparison_{dataset_name}.png')
    plt.close()

def main():
    """Main function to orchestrate the analysis"""
    # Create output directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Load data
    grind_data, brew_data = load_data()
    
    # Perform exploratory data analysis
    exploratory_data_analysis(grind_data, brew_data)
    
    # Correlation analysis
    grind_corr = correlation_analysis(grind_data, 'Grind', 'figures/grind_correlation.png')
    brew_corr = correlation_analysis(brew_data, 'Brew', 'figures/brew_correlation.png')
    
    # Preprocess data for modeling
    X_grind, y_grind = preprocess_grind_data(grind_data)
    
    # Train and evaluate models on grind data
    if X_grind is not None and y_grind is not None:
        xgb_model = run_xgboost_model(X_grind, y_grind, 'Grind')
        cat_model = run_catboost_model(X_grind, y_grind, 'Grind')
        
        # Compare models
        try:
            from sklearn.model_selection import KFold
            compare_models(xgb_model, cat_model, X_grind, y_grind, 'Grind')
        except Exception as e:
            print(f"Error in model comparison: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 