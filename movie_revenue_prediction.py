#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import shap
import re
import ast
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
# Loading data
def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loading completed, {df.shape[0]} rows in total，{df.shape[1]} features")
    return df

# Data preprocessing
def preprocess_data(df, budget_threshold=15000000):
    print("Data preprocessing...")
    
    # Copy data to avoid modification of raw data
    df = df.copy()
    
    # Convert budget and income to numerical types
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    
    # Keep effective budget and revenue data
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    
    # Keep low-budget movie data
    df_low_budget = df[df['budget'] <= budget_threshold].copy()
    print(f"Keep movies with budget less than {budget_threshold/1000000}，{df_low_budget.shape[0]} records in total")
    
    # Process release date
    df_low_budget['release_date'] = pd.to_datetime(df_low_budget['release_date'], errors='coerce')
    df_low_budget['release_year'] = df_low_budget['release_date'].dt.year
    df_low_budget['release_month'] = df_low_budget['release_date'].dt.month
    
    # Calculate ROI (Return on Investment)
    df_low_budget['roi'] = (df_low_budget['revenue'] - df_low_budget['budget']) / df_low_budget['budget']
    
    # Features of list types such as genres, production_companies
    list_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
    for col in list_columns:
        df_low_budget[col] = df_low_budget[col].fillna('[]')
        
    print("Preprocessing Completed")
    return df_low_budget
# Feature engineering
def feature_engineering(df):
    print("Start Feature Engineering...")
    
    # Extract category features
    df['genres_count'] = df['genres'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['spoken_languages_count'] = df['spoken_languages'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['keywords_count'] = df['keywords'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    
    # Extract main categories
    def extract_main_category(x, category_name):
        try:
            categories = str(x).split(',')
            if len(categories) > 0:
                return categories[0].strip()
            return 'Unknown'
        except:
            return 'Unknown'
    
    df['main_genre'] = df['genres'].apply(lambda x: extract_main_category(x, 'genres'))
    df['main_production_company'] = df['production_companies'].apply(lambda x: extract_main_category(x, 'production_companies'))
    df['main_production_country'] = df['production_countries'].apply(lambda x: extract_main_category(x, 'production_countries'))
    
    # Extract runtime
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # Create feature is_english
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    
    # Extract vote features
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    
    # Create a combination of scores and votes
    df['vote_score'] = df['vote_average'] * np.log1p(df['vote_count'])
    
    # Missing values
    numeric_features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'vote_score']
    for feature in numeric_features:
        df[feature] = df[feature].fillna(df[feature].median())
    
    print("Feature Engineering Completed")
    return df

# Prepare Training Data
def prepare_model_data(df):
    print("Prepare Training Data...")
    
    # Feature Selection
    features = [
        'budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'vote_score',
        'genres_count', 'production_companies_count', 'production_countries_count',
        'spoken_languages_count', 'keywords_count', 'release_year', 'release_month',
        'is_english', 'main_genre', 'main_production_company', 'main_production_country'
    ]
    
    # Target variable
    target = 'revenue'
    
    # Features and target variables
    X = df[features].copy()
    y = np.log1p(df[target])  # A logarithmic conversion of revenue to bring its distribution closer to normal
    
    # split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    print(f"train set: {X_train.shape[0]}, validation set: {X_val.shape[0]}, test set:: {X_test.shape[0]}")
    
    # Get feature names
    numeric_features = ['budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'vote_score',
                       'genres_count', 'production_companies_count', 'production_countries_count',
                       'spoken_languages_count', 'keywords_count', 'release_year', 'release_month', 'is_english']
    categorical_features = ['main_genre', 'main_production_company', 'main_production_country']
    
    return X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
# Pipeline
def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Train and evaluate traditional models
def train_evaluate_traditional_models(X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features):
    print("Train and evaluate traditional models...")
    
    # Building preprocessing pipes
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # Define the models
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate traditional models
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Train {name} Model...")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # evaluate on validation set
        y_val_pred = pipeline.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # evaluate on test set
        y_test_pred = pipeline.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # save the results
        results[name] = {
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        
        trained_models[name] = pipeline
        
        print(f"{name} Evaluation Result:")
        print(f"  validation RMSE: {val_rmse:.4f}")
        print(f"  validation MAE: {val_mae:.4f}")
        print(f"  validation R²: {val_r2:.4f}")
        print(f"  test RMSE: {test_rmse:.4f}")
        print(f"  test MAE: {test_mae:.4f}")
        print(f"  test R²: {test_r2:.4f}")
    
    return results, trained_models

# Train and evaluate TabNet model
def train_evaluate_tabnet(X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features):
    print("Train and evaluate TabNet model...")
    
    # Building preprocessing pipes
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # Preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # Convert to required TabNet Format
    X_train_processed = X_train_processed.astype(np.float32)
    X_val_processed = X_val_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    y_train = y_train.values.reshape(-1, 1).astype(np.float32) if hasattr(y_train, "values") else y_train.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1).astype(np.float32) if hasattr(y_val, "values") else y_val.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1).astype(np.float32) if hasattr(y_test, "values") else y_test.reshape(-1, 1)

    # Define the TabNet model
    tabnet_model = TabNetRegressor(
        n_d=8,  # 决策步骤的维度
        n_a=8,  # 注意力步骤的维度
        n_steps=3,  # 决策步骤的数量
        gamma=1.5,  # 稀疏性参数
        n_independent=2,  # 独立层的数量
        n_shared=2,  # 共享层的数量
        momentum=0.3,  # 动量参数
        mask_type='entmax',  # 掩码类型
        lambda_sparse=1e-3,  # 稀疏性正则化参数
        optimizer_fn=torch.optim.Adam,  # 优化器
        optimizer_params=dict(lr=2e-2),  # 优化器参数
        scheduler_params={"step_size":10, "gamma":0.9},  # 学习率调度器参数
        scheduler_fn=torch.optim.lr_scheduler.StepLR,  # 学习率调度器
        device_name='auto',  # 设备名称
        verbose=1  # 输出详细程度
    )
    
    # Train the model
    tabnet_model.fit(
        X_train=X_train_processed, y_train=y_train,
        eval_set=[(X_val_processed, y_val)],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # Evaluate on validation set
    y_val_pred = tabnet_model.predict(X_val_processed)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # Evaluate on test set
    y_test_pred = tabnet_model.predict(X_test_processed)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # save the results
    results = {
        'TabNet': {
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }
    
    print("TabNet Evaluation Result:")
    print(f"  validation RMSE: {val_rmse:.4f}")
    print(f"  validation MAE: {val_mae:.4f}")
    print(f"  validation R²: {val_r2:.4f}")
    print(f"  test RMSE: {test_rmse:.4f}")
    print(f"  test MAE: {test_mae:.4f}")
    print(f"  test R²: {test_r2:.4f}")
    
    # Get feature importance
    feature_importances = tabnet_model.feature_importances_
    
    return results, tabnet_model, preprocessor, feature_importances

# Visualize
def visualize_model_comparison(traditional_results, tabnet_results):
    print("Visualize the comparison results...")
    
    # merge the result
    all_results = {**traditional_results, **tabnet_results}
    
    # Extract the metrics
    models = list(all_results.keys())
    rmse_values = [all_results[model]['test_rmse'] for model in models]
    mae_values = [all_results[model]['test_mae'] for model in models]
    r2_values = [all_results[model]['test_r2'] for model in models]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE
    axes[0].bar(models, rmse_values, color='skyblue')
    axes[0].set_title('Testset RMSE')
    axes[0].set_ylabel('RMSE')
    axes[0].set_xticklabels(models, rotation=45)
    
    # MAE
    axes[1].bar(models, mae_values, color='lightgreen')
    axes[1].set_title('Testset MAE')
    axes[1].set_ylabel('MAE')
    axes[1].set_xticklabels(models, rotation=45)
    
    # R²
    axes[2].bar(models, r2_values, color='salmon')
    axes[2].set_title('Testset R²')
    axes[2].set_ylabel('R^2')
    axes[2].set_xticklabels(models, rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    print("Model comparison results saved as 'model_comparison.png'")

# Visualize the feature importance
def visualize_feature_importance(model, preprocessor, X_train, feature_names=None):
    print("Visualize the feature importance...")
    
    if 'RandomForest' in str(model) or 'XGBoost' in str(model):
        # Get feature names
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(preprocessor.transform(X_train.iloc[[0]]).shape[1])]
        
        # get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = model[-1].feature_importances_
        
        # create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)
        
        # visualize
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('xgboost_feature importance.png', dpi=300)
        plt.close()
        
        print("Feature Importance saved as 'xgboost_feature importance.png'")

# Visualize TabNet feature importance
def visualize_tabnet_feature_importance(feature_importances, preprocessor, numeric_features, categorical_features):
    print("Visualize TabNet feature importance...")
    
    # Get feature names
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
    
    # create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)
    
    # visualize
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('TabNet Feature Importance')
    plt.tight_layout()
    plt.savefig('tabnet_feature_importance.png', dpi=300)
    plt.close()
    
    print("TabNet Feature Importance saved as 'tabnet_feature_importance.png'")

# Visualize the Prediction Results
def visualize_predictions(y_test, y_pred, model_name):
    print(f"Visualize {model_name} Prediction Results...")
    
    # convert back to original scale
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)
    
    # plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_exp, y_pred_exp, alpha=0.5)
    plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title(f'{model_name} Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=300)
    plt.close()
    
    print(f"{model_name} Prediction Results saved as '{model_name}_predictions.png'")

# Residual Analysis
def visualize_residuals(y_test, y_pred, model_name):
    print(f"Visualize {model_name} Residual Analysis...")
    
    # Calculate
    residuals = y_test - y_pred
    
    # ploy
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residuals')
    plt.title('Residual Scatter Plot')
    
    # Residuals Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_residuals.png', dpi=300)
    plt.close()
    
    print(f"{model_name} Residual Analysis saved as '{model_name}_residuals.png'")

# main
def main():
    # Loading data
    df = load_data('TMDB_movie_dataset_v11.csv')
    
    # Preprocessing
    df_low_budget = preprocess_data(df)
    
    # Feature Engineering
    df_processed = feature_engineering(df_low_budget)
    
    # split
    X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features = prepare_model_data(df_processed)
    
    # train and evaluate traditional models
    traditional_results, traditional_models = train_evaluate_traditional_models(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # train and evaluate TabNet
    tabnet_results, tabnet_model, tabnet_preprocessor, feature_importances = train_evaluate_tabnet(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # visualize model comparison
    visualize_model_comparison(traditional_results, tabnet_results)
    
    # visualize feature importance
    for name, model in traditional_models.items():
        if name in ['RandomForest', 'XGBoost']:
            visualize_feature_importance(model, model['preprocessor'], X_train)
    
    # visualize tabnet feature importance
    visualize_tabnet_feature_importance(feature_importances, tabnet_preprocessor, numeric_features, categorical_features)
    
    # Best model result
    best_model_name = max(traditional_results, key=lambda x: traditional_results[x]['test_r2'])
    best_model = traditional_models[best_model_name]
    y_test_pred = best_model.predict(X_test)
    visualize_predictions(y_test, y_test_pred, best_model_name)
    visualize_residuals(y_test, y_test_pred, best_model_name)
    
    # TabNet result
    y_test_pred_tabnet = tabnet_model.predict(tabnet_preprocessor.transform(X_test))
    # TabNet prep reshape
    y_test_pred_tabnet = np.ravel(y_test_pred_tabnet)

    visualize_predictions(y_test, y_test_pred_tabnet, 'TabNet')
    visualize_residuals(y_test, y_test_pred_tabnet, 'TabNet')
    
    print("done")

if __name__ == "__main__":
    main()