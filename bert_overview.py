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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False  
# 加载数据
def load_data(file_path):
    """
    加载电影数据集
    
    参数:
    file_path: 数据集文件路径
    
    返回:
    df: 加载的数据集
    """
    print("加载数据集...")
    df = pd.read_csv(file_path)
    print(f"数据集加载完成，共 {df.shape[0]} 条记录，{df.shape[1]} 个特征")
    return df

# 数据预处理
def preprocess_data(df, budget_threshold=15000000):
    """
    数据预处理，包括筛选低成本电影、处理缺失值、转换数据类型等
    
    参数:
    df: 原始数据集
    budget_threshold: 预算阈值，默认为1500万美元
    
    返回:
    df_low_budget: 处理后的低成本电影数据集
    """
    print("开始数据预处理...")
    
    # 复制数据，避免修改原始数据
    df = df.copy()
    
    # 将预算和收入转换为数值类型
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    
    # 筛选有效的预算和收入数据
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]
    
    # 筛选低成本电影
    df_low_budget = df[df['budget'] <= budget_threshold].copy()
    print(f"筛选出预算低于 {budget_threshold/1000000} 百万美元的电影，共 {df_low_budget.shape[0]} 条记录")
    
    # 处理发行日期
    df_low_budget['release_date'] = pd.to_datetime(df_low_budget['release_date'], errors='coerce')
    df_low_budget['release_year'] = df_low_budget['release_date'].dt.year
    df_low_budget['release_month'] = df_low_budget['release_date'].dt.month
    
    # 计算ROI (Return on Investment)
    df_low_budget['roi'] = (df_low_budget['revenue'] - df_low_budget['budget']) / df_low_budget['budget']
    
    # 处理列表类型的特征（如genres, production_companies等）
    list_columns = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
    for col in list_columns:
        df_low_budget[col] = df_low_budget[col].fillna('[]')
        
    print("数据预处理完成")
    return df_low_budget
# 特征工程
def feature_engineering(df):
    """
    特征工程，包括提取类别特征、文本特征等
    
    参数:
    df: 预处理后的数据集
    
    返回:
    df: 增强特征后的数据集
    """
    print("开始特征工程...")
    
    # 提取类别特征数量
    df['genres_count'] = df['genres'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['production_companies_count'] = df['production_companies'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['production_countries_count'] = df['production_countries'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['spoken_languages_count'] = df['spoken_languages'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['keywords_count'] = df['keywords'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    # 提取主要类别
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
    
    # 提取电影时长特征
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    
    # 创建是否为英语电影的特征
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    
    # 提取评分特征
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    
    # 创建评分与投票数的组合特征
    df['vote_score'] = df['vote_average'] * np.log1p(df['vote_count'])
    
    # 处理缺失值
    numeric_features = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'vote_score']
    for feature in numeric_features:
        df[feature] = df[feature].fillna(df[feature].median())
    
    df['overview'] = df['overview'].fillna('')
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    overview_embeddings = bert_model.encode(df['overview'].tolist(), show_progress_bar=True)
    for i in range(overview_embeddings.shape[1]):
        df[f'overview_emb_{i}'] = overview_embeddings[:, i]

    print("特征工程完成")
    return df

# 准备模型训练数据
def prepare_model_data(df):
    """
    准备模型训练数据，包括特征选择、数据分割等
    
    参数:
    df: 特征工程后的数据集
    
    返回:
    X_train, X_val, X_test, y_train, y_val, y_test: 训练集、验证集和测试集
    feature_names: 特征名称列表
    """
    print("准备模型训练数据...")
    
    # 选择特征
    features = [
        'budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'vote_score',
        'genres_count', 'production_companies_count', 'production_countries_count',
        'spoken_languages_count', 'keywords_count', 'release_year', 'release_month',
        'is_english', 'main_genre', 'main_production_company', 'main_production_country'
    ]
    
    bert_features = [col for col in df.columns if col.startswith('overview_emb_')]
    features = features + bert_features

    # 选择目标变量
    target = 'revenue'
    
    # 准备特征和目标变量
    X = df[features].copy()
    y = np.log1p(df[target])  # 对收入进行对数转换，使其分布更接近正态
    
    # 分割数据集为训练集、验证集和测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    
    print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 获取特征名称
    numeric_features = ['budget', 'runtime', 'popularity', 'vote_average', 'vote_count', 'vote_score',
                       'genres_count', 'production_companies_count', 'production_countries_count',
                       'spoken_languages_count', 'keywords_count', 'release_year', 'release_month', 'is_english'] + bert_features
    categorical_features = ['main_genre', 'main_production_company', 'main_production_country']
    
    return X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
# 构建预处理管道
def build_preprocessor(numeric_features, categorical_features):
    """
    构建特征预处理管道
    
    参数:
    numeric_features: 数值型特征列表
    categorical_features: 类别型特征列表
    
    返回:
    preprocessor: 预处理管道
    """
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

# 训练和评估传统机器学习模型
def train_evaluate_traditional_models(X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features):
    """
    训练和评估传统机器学习模型
    
    参数:
    X_train, X_val, X_test, y_train, y_val, y_test: 训练集、验证集和测试集
    numeric_features, categorical_features: 数值型和类别型特征列表
    
    返回:
    results: 模型评估结果
    models: 训练好的模型
    """
    print("训练和评估传统机器学习模型...")
    
    # 构建预处理管道
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 定义模型
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    # 训练和评估模型
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"训练 {name} 模型...")
        
        # 构建完整管道
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # 训练模型
        pipeline.fit(X_train, y_train)
        
        # 在验证集上评估
        y_val_pred = pipeline.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # 在测试集上评估
        y_test_pred = pipeline.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 保存结果
        results[name] = {
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
        
        trained_models[name] = pipeline
        
        print(f"{name} 模型评估结果:")
        print(f"  验证集 RMSE: {val_rmse:.4f}")
        print(f"  验证集 MAE: {val_mae:.4f}")
        print(f"  验证集 R²: {val_r2:.4f}")
        print(f"  测试集 RMSE: {test_rmse:.4f}")
        print(f"  测试集 MAE: {test_mae:.4f}")
        print(f"  测试集 R²: {test_r2:.4f}")
    
    return results, trained_models

# 训练和评估TabNet模型
def train_evaluate_tabnet(X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features):
    """
    训练和评估TabNet模型
    
    参数:
    X_train, X_val, X_test, y_train, y_val, y_test: 训练集、验证集和测试集
    numeric_features, categorical_features: 数值型和类别型特征列表
    
    返回:
    results: 模型评估结果
    model: 训练好的模型
    """
    print("训练和评估TabNet模型...")
    
    # 构建预处理管道
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 预处理数据
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    # 转换为TabNet所需的格式
    X_train_processed = X_train_processed.astype(np.float32)
    X_val_processed = X_val_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_val = y_val.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    y_train = y_train.values.reshape(-1, 1).astype(np.float32) if hasattr(y_train, "values") else y_train.reshape(-1, 1)
    y_val = y_val.values.reshape(-1, 1).astype(np.float32) if hasattr(y_val, "values") else y_val.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1).astype(np.float32) if hasattr(y_test, "values") else y_test.reshape(-1, 1)

    # 定义TabNet模型
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
    
    # 训练模型
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
    
    # 在验证集上评估
    y_val_pred = tabnet_model.predict(X_val_processed)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    # 在测试集上评估
    y_test_pred = tabnet_model.predict(X_test_processed)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 保存结果
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
    
    print("TabNet 模型评估结果:")
    print(f"  验证集 RMSE: {val_rmse:.4f}")
    print(f"  验证集 MAE: {val_mae:.4f}")
    print(f"  验证集 R²: {val_r2:.4f}")
    print(f"  测试集 RMSE: {test_rmse:.4f}")
    print(f"  测试集 MAE: {test_mae:.4f}")
    print(f"  测试集 R²: {test_r2:.4f}")
    
    # 获取特征重要性
    feature_importances = tabnet_model.feature_importances_
    
    return results, tabnet_model, preprocessor, feature_importances

# 可视化模型比较结果
def visualize_model_comparison(traditional_results, tabnet_results):
    """
    可视化不同模型的比较结果
    
    参数:
    traditional_results: 传统模型的评估结果
    tabnet_results: TabNet模型的评估结果
    """
    print("可视化模型比较结果...")
    
    # 合并结果
    all_results = {**traditional_results, **tabnet_results}
    
    # 提取测试集上的评估指标
    models = list(all_results.keys())
    rmse_values = [all_results[model]['test_rmse'] for model in models]
    mae_values = [all_results[model]['test_mae'] for model in models]
    r2_values = [all_results[model]['test_r2'] for model in models]
    
    # 创建图形
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
    
    print("模型比较结果已保存为 'model_comparison.png'")

# 可视化特征重要性
def visualize_feature_importance(model, preprocessor, X_train, feature_names=None):
    """
    可视化特征重要性
    
    参数:
    model: 训练好的模型
    preprocessor: 特征预处理器
    X_train: 训练数据
    feature_names: 特征名称列表
    """
    print("可视化特征重要性...")
    
    if 'RandomForest' in str(model) or 'XGBoost' in str(model):
        # 获取特征名称
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(preprocessor.transform(X_train.iloc[[0]]).shape[1])]
        
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = model[-1].feature_importances_
        
        # 创建特征重要性数据框
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)
        
        # 可视化
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        plt.savefig('xgboost_feature importance.png', dpi=300)
        plt.close()
        
        print("特征重要性已保存为 'xgboost_feature importance.png'")

# 可视化TabNet特征重要性
def visualize_tabnet_feature_importance(feature_importances, preprocessor, numeric_features, categorical_features):
    """
    可视化TabNet特征重要性
    
    参数:
    feature_importances: TabNet模型的特征重要性
    preprocessor: 特征预处理器
    numeric_features: 数值型特征列表
    categorical_features: 类别型特征列表
    """
    print("可视化TabNet特征重要性...")
    
    # 获取特征名称
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
    
    # 创建特征重要性数据框
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # 按重要性排序
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(20)
    
    # 可视化
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('TabNet Feature Importance')
    plt.tight_layout()
    plt.savefig('tabnet_feature_importance.png', dpi=300)
    plt.close()
    
    print("TabNet特征重要性已保存为 'tabnet_feature_importance.png'")

# 可视化预测结果
def visualize_predictions(y_test, y_pred, model_name):
    """
    可视化预测结果
    
    参数:
    y_test: 测试集真实值
    y_pred: 测试集预测值
    model_name: 模型名称
    """
    print(f"可视化 {model_name} 预测结果...")
    
    # 转换回原始尺度
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)
    
    # 创建散点图
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
    
    print(f"{model_name} 预测结果已保存为 '{model_name}_predictions.png'")

# 可视化残差分析
def visualize_residuals(y_test, y_pred, model_name):
    """
    可视化残差分析
    
    参数:
    y_test: 测试集真实值
    y_pred: 测试集预测值
    model_name: 模型名称
    """
    print(f"可视化 {model_name} 残差分析...")
    
    # 计算残差
    residuals = y_test - y_pred
    
    # 创建残差图
    plt.figure(figsize=(12, 6))
    
    # 残差散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Value')
    plt.ylabel('Residuals')
    plt.title('Residual Scatter Plot')
    
    # 残差分布图
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_residuals.png', dpi=300)
    plt.close()
    
    print(f"{model_name} 残差分析已保存为 '{model_name}_residuals.png'")

# 主函数
def main():
    """
    主函数，执行完整的数据处理和模型训练流程
    """
    # 加载数据
    df = load_data('TMDB_movie_dataset_v11.csv')
    
    # 数据预处理
    df_low_budget = preprocess_data(df)
    
    # 特征工程
    df_processed = feature_engineering(df_low_budget)
    
    # 准备模型训练数据
    X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features = prepare_model_data(df_processed)
    
    # 训练和评估传统机器学习模型
    traditional_results, traditional_models = train_evaluate_traditional_models(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # 训练和评估TabNet模型
    tabnet_results, tabnet_model, tabnet_preprocessor, feature_importances = train_evaluate_tabnet(
        X_train, X_val, X_test, y_train, y_val, y_test, numeric_features, categorical_features
    )
    
    # 可视化模型比较结果
    visualize_model_comparison(traditional_results, tabnet_results)
    
    # 可视化特征重要性
    for name, model in traditional_models.items():
        if name in ['RandomForest', 'XGBoost']:
            visualize_feature_importance(model, model['preprocessor'], X_train)
    
    # 可视化TabNet特征重要性
    visualize_tabnet_feature_importance(feature_importances, tabnet_preprocessor, numeric_features, categorical_features)
    
    # 可视化最佳模型的预测结果
    best_model_name = max(traditional_results, key=lambda x: traditional_results[x]['test_r2'])
    best_model = traditional_models[best_model_name]
    y_test_pred = best_model.predict(X_test)
    visualize_predictions(y_test, y_test_pred, best_model_name)
    visualize_residuals(y_test, y_test_pred, best_model_name)
    
    # 可视化TabNet的预测结果
    y_test_pred_tabnet = tabnet_model.predict(tabnet_preprocessor.transform(X_test))
    # TabNet预测值 reshape 成一维，避免残差图报错
    y_test_pred_tabnet = np.ravel(y_test_pred_tabnet)

    visualize_predictions(y_test, y_test_pred_tabnet, 'TabNet')
    visualize_residuals(y_test, y_test_pred_tabnet, 'TabNet')
    
    print("done")

if __name__ == "__main__":
    main()