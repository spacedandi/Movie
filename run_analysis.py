#!/usr/bin/env python
# coding: utf-8

# Run Low-Budget Movie Revenue Prediction Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from movie_revenue_prediction import *
import os

# Create output directories
if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('results'):
    os.makedirs('results')

# Set plot parameters
plt.rcParams['font.sans-serif'] = ['Arial']  
plt.rcParams['axes.unicode_minus'] = False  

def run_analysis():
    """
    Run the complete analysis pipeline, including data processing, model training, and result visualization
    """
    print("Starting low-budget movie revenue prediction analysis...")
    
    # Load data
    df = load_data('TMDB_movie_dataset_v11.csv')
    
    # Data preprocessing
    df_low_budget = preprocess_data(df)
    
    # Save low-budget movie dataset
    df_low_budget.to_csv('results/low_budget_movies.csv', index=False)
    
    # Descriptive statistical analysis
    perform_descriptive_analysis(df_low_budget)
    
    # Feature engineering
    df_processed = feature_engineering(df_low_budget)
    
    # Prepare model training data
    X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features = prepare_model_data(df_processed)
    
    # Train and evaluate traditional machine learning models
    traditional_results, traditional_models = train_evaluate_traditional_models(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # Save traditional model results
    save_model_results(traditional_results, 'results/traditional_model_results.csv')
    
    # Train and evaluate TabNet model
    tabnet_results, tabnet_model, tabnet_preprocessor, feature_importances = train_evaluate_tabnet(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # Save TabNet model results
    save_model_results(tabnet_results, 'results/tabnet_model_results.csv')
    
    # Visualize model comparison results
    visualize_model_comparison(traditional_results, tabnet_results)
    
    # Visualize feature importance
    for name, model in traditional_models.items():
        if name in ['RandomForest', 'XGBoost']:
            visualize_feature_importance(model, model['preprocessor'], X_train)
    
    # Visualize TabNet feature importance
    visualize_tabnet_feature_importance(feature_importances, tabnet_preprocessor, numeric_features, categorical_features)
    
    # Visualize best model prediction results
    best_model_name = max(traditional_results, key=lambda x: traditional_results[x]['test_r2'])
    best_model = traditional_models[best_model_name]
    y_test_pred = best_model.predict(X_test)
    visualize_predictions(y_test, y_test_pred, best_model_name)
    visualize_residuals(y_test, y_test_pred, best_model_name)
    
    # Visualize TabNet prediction results
    X_test_tabnet = tabnet_preprocessor.transform(X_test)
    y_test_pred_tabnet = tabnet_model.predict(X_test_tabnet)

    # 保证是一维，避免残差计算出错
    y_test_pred_tabnet = np.ravel(y_test_pred_tabnet)

    visualize_predictions(y_test, y_test_pred_tabnet, 'TabNet')
    visualize_residuals(y_test, y_test_pred_tabnet, 'TabNet')
    
    # Compare low-budget movie models with all-movie models
    compare_low_budget_vs_all(df)
    
    print("Analysis complete, results saved to 'results' and 'figures' directories")

def perform_descriptive_analysis(df):
    """
    Perform descriptive statistical analysis
    
    Parameters:
    df: Low-budget movie dataset
    """
    print("Performing descriptive statistical analysis...")
    
    # Basic statistics
    desc_stats = df[['budget', 'revenue', 'roi', 'runtime', 'vote_average', 'vote_count', 'popularity']].describe()
    desc_stats.to_csv('results/descriptive_statistics.csv')
    
    # Budget distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['budget'], bins=30, kde=True)
    plt.title('Low-Budget Movie Budget Distribution')
    plt.xlabel('Budget (USD)')
    plt.ylabel('Frequency')
    plt.savefig('figures/budget_distribution.png', dpi=300)
    plt.close()
    
    # Revenue distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['revenue'], bins=30, kde=True)
    plt.title('Low-Budget Movie Revenue Distribution')
    plt.xlabel('Revenue (USD)')
    plt.ylabel('Frequency')
    plt.savefig('figures/revenue_distribution.png', dpi=300)
    plt.close()
    
    # Log revenue distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(np.log1p(df['revenue']), bins=30, kde=True)
    plt.title('Low-Budget Movie Log Revenue Distribution')
    plt.xlabel('Log Revenue')
    plt.ylabel('Frequency')
    plt.savefig('figures/log_revenue_distribution.png', dpi=300)
    plt.close()
    
    # Relationship between budget and revenue
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='budget', y='revenue', data=df, alpha=0.5)
    plt.title('Relationship Between Budget and Revenue')
    plt.xlabel('Budget (USD)')
    plt.ylabel('Revenue (USD)')
    plt.savefig('figures/budget_vs_revenue.png', dpi=300)
    plt.close()
    
    # Relationship between budget and ROI
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='budget', y='roi', data=df, alpha=0.5)
    plt.title('Relationship Between Budget and Return on Investment')
    plt.xlabel('Budget (USD)')
    plt.ylabel('Return on Investment (ROI)')
    plt.ylim(-1, 10)  # Limit y-axis range to exclude extreme values
    plt.savefig('figures/budget_vs_roi.png', dpi=300)
    plt.close()
    
    # Statistics by year: movie count and average revenue
    yearly_stats = df.groupby('release_year').agg({
        'id': 'count',
        'revenue': 'mean',
        'budget': 'mean',
        'roi': 'mean'
    }).reset_index()
    yearly_stats.columns = ['Year', 'Movie Count', 'Average Revenue', 'Average Budget', 'Average ROI']
    yearly_stats = yearly_stats[yearly_stats['Year'] >= 2000]  # Focus only on data after 2000
    yearly_stats.to_csv('results/yearly_statistics.csv', index=False)
    
    # Visualize yearly trends
    plt.figure(figsize=(12, 8))
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Movie Count', color='tab:blue')
    ax1.plot(yearly_stats['Year'], yearly_stats['Movie Count'], color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Revenue (Million USD)', color='tab:red')
    ax2.plot(yearly_stats['Year'], yearly_stats['Average Revenue']/1000000, color='tab:red', marker='s')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Low-Budget Movie Yearly Trends')
    fig.tight_layout()
    plt.savefig('figures/yearly_trends.png', dpi=300)
    plt.close()
    
    # Statistics by genre
    df['main_genre'] = df['genres'].apply(lambda x: str(x).split(',')[0].strip() if pd.notna(x) and len(str(x).split(',')) > 0 else 'Unknown')
    genre_stats = df.groupby('main_genre').agg({
        'id': 'count',
        'revenue': 'mean',
        'budget': 'mean',
        'roi': 'mean'
    }).reset_index()
    genre_stats.columns = ['Genre', 'Movie Count', 'Average Revenue', 'Average Budget', 'Average ROI']
    genre_stats = genre_stats.sort_values('Movie Count', ascending=False).head(10)  # Focus only on top 10 genres
    genre_stats.to_csv('results/genre_statistics.csv', index=False)
    
    # Visualize genre statistics
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Genre', y='Average Revenue', data=genre_stats)
    plt.title('Average Revenue of Low-Budget Movies by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average Revenue (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/genre_revenue.png', dpi=300)
    plt.close()
    
    print("Descriptive statistical analysis completed")

def save_model_results(results, file_path):
    """
    Save model evaluation results
    
    Parameters:
    results: Model evaluation results
    file_path: Save path
    """
    # Create results dataframe
    results_df = pd.DataFrame(columns=['Model', 'Val_RMSE', 'Val_MAE', 'Val_R2', 'Test_RMSE', 'Test_MAE', 'Test_R2'])
    
    for i, (model_name, metrics) in enumerate(results.items()):
        results_df.loc[i] = [
            model_name,
            metrics['val_rmse'],
            metrics['val_mae'],
            metrics['val_r2'],
            metrics['test_rmse'],
            metrics['test_mae'],
            metrics['test_r2']
        ]
    
    # Save results
    results_df.to_csv(file_path, index=False)

def compare_low_budget_vs_all(df):
    """
    Compare low-budget movie models with all-movie models
    
    Parameters:
    df: Original movie dataset
    """
    print("Comparing low-budget movie models with all-movie models...")
    
    # Preprocess all movie data to generate release_year and release_month
    df_all = preprocess_data(df)

    # Filter low-budget movies after preprocessing
    df_low = df_all[df_all['budget'] <= 15000000].copy()

    # Feature engineering
    df_all = feature_engineering(df_all)
    df_low = feature_engineering(df_low)
    
    # Prepare all movie data
    features = [
        'budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'vote_score',
        'genres_count', 'production_companies_count', 'production_countries_count',
        'spoken_languages_count', 'keywords_count', 'release_year', 'release_month',
        'is_english'
    ]
    
    X_all = df_all[features].copy()
    y_all = np.log1p(df_all['revenue'])
    
    X_low = df_low[features].copy()
    y_low = np.log1p(df_low['revenue'])
    
    # Split data
    X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    # Evaluation results
    comparison_results = pd.DataFrame(columns=['Model', 'Dataset', 'RMSE', 'MAE', 'R2'])
    row = 0
    
    for name, model in models.items():
        print(f"Training {name} model...")
        
        # Train on all movie data
        model_all = model.fit(X_train_all, y_train_all)
        y_pred_all = model_all.predict(X_test_all)
        rmse_all = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
        mae_all = mean_absolute_error(y_test_all, y_pred_all)
        r2_all = r2_score(y_test_all, y_pred_all)
        
        comparison_results.loc[row] = [name, 'All Movies', rmse_all, mae_all, r2_all]
        row += 1
        
        # Train on low-budget movie data
        model_low = model.fit(X_train_low, y_train_low)
        y_pred_low = model_low.predict(X_test_low)
        rmse_low = np.sqrt(mean_squared_error(y_test_low, y_pred_low))
        mae_low = mean_absolute_error(y_test_low, y_pred_low)
        r2_low = r2_score(y_test_low, y_pred_low)
        
        comparison_results.loc[row] = [name, 'Low Budget Movies', rmse_low, mae_low, r2_low]
        row += 1
        
        # Train on all movie data, but test on low-budget movie data
        y_pred_cross = model_all.predict(X_test_low)
        rmse_cross = np.sqrt(mean_squared_error(y_test_low, y_pred_cross))
        mae_cross = mean_absolute_error(y_test_low, y_pred_cross)
        r2_cross = r2_score(y_test_low, y_pred_cross)
        
        comparison_results.loc[row] = [name, 'All -> Low Budget', rmse_cross, mae_cross, r2_cross]
        row += 1
    
    # Save comparison results
    comparison_results.to_csv('results/model_comparison_all_vs_low.csv', index=False)
    
    # Visualize comparison results
    plt.figure(figsize=(15, 6))
    
    # RMSE comparison
    plt.subplot(1, 3, 1)
    sns.barplot(x='Model', y='RMSE', hue='Dataset', data=comparison_results)
    plt.title('RMSE Comparison')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # MAE comparison
    plt.subplot(1, 3, 2)
    sns.barplot(x='Model', y='MAE', hue='Dataset', data=comparison_results)
    plt.title('MAE Comparison')
    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    # R2 comparison
    plt.subplot(1, 3, 3)
    sns.barplot(x='Model', y='R2', hue='Dataset', data=comparison_results)
    plt.title('R² Comparison')
    plt.xlabel('Model')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('figures/all_vs_low_budget_comparison.png', dpi=300)
    plt.close()
    
    print("Comparison completed")

if __name__ == "__main__":
    run_analysis()