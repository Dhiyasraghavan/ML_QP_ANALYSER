"""
NEET Analyzer - Enhanced Model Training Suite v3.0
Complete AI-powered training pipeline with advanced features
"""

import pandas as pd
import numpy as np
import warnings
import os
import time
from pathlib import Path

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,
    ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import (
    TimeSeriesSplit, train_test_split, GridSearchCV, 
    cross_val_score, validation_curve
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, mean_squared_error, r2_score,
    mean_absolute_error, accuracy_score, f1_score, precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Advanced model validation
from sklearn.model_selection import learning_curve, cross_validate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Save/load models
from joblib import dump, load

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class EnhancedNEETModelTrainer:
    """
    Advanced NEET model trainer with comprehensive feature engineering,
    robust validation, and enhanced ensemble methods
    """
    
    def __init__(self, random_state=42, verbose=True):
        self.random_state = random_state
        self.verbose = verbose
        self.models_trained = {}
        self.feature_importance = {}
        self.training_history = {}
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        
    def log(self, message, level="INFO"):
        """Enhanced logging with timestamps"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            prefixes = {
                "INFO": "ℹ️",
                "SUCCESS": "✅", 
                "WARNING": "⚠️",
                "ERROR": "❌",
                "PROGRESS": "🔄"
            }
            prefix = prefixes.get(level, "📋")
            print(f"[{timestamp}] {prefix} {message}")

    def validate_data_quality(self, data, required_columns, data_type="dataset"):
        """Comprehensive data quality validation"""
        self.log(f"🔍 Validating {data_type} quality...", "PROGRESS")
        
        issues = []
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for sufficient data
        if len(data) < 20:
            issues.append(f"Insufficient data: only {len(data)} rows")
        
        # Check for data types and ranges
        if 'year' in data.columns:
            invalid_years = data[(data['year'] < 2010) | (data['year'] > 2030)]
            if len(invalid_years) > 0:
                issues.append(f"Invalid years found: {invalid_years['year'].tolist()}")
        
        # Check for extreme outliers
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns:
                q1, q3 = data[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = data[(data[col] < q1 - 3*iqr) | (data[col] > q3 + 3*iqr)]
                if len(outliers) > len(data) * 0.1:  # More than 10% outliers
                    issues.append(f"High outlier rate in {col}: {len(outliers)} outliers")
        
        if issues:
            self.log(f"⚠️ Data quality issues found in {data_type}:", "WARNING")
            for issue in issues:
                self.log(f"   - {issue}", "WARNING")
        else:
            self.log(f"✅ {data_type} quality validation passed", "SUCCESS")
        
        return len(issues) == 0, issues

    def enhanced_text_preprocessing(self, data):
        """Advanced text preprocessing with domain-specific optimizations"""
        self.log("🔄 Applying enhanced text preprocessing...", "PROGRESS")
        
        # Remove extremely short questions (likely parsing errors)
        original_len = len(data)
        data = data[data['question'].str.len() > 10].copy()
        removed = original_len - len(data)
        if removed > 0:
            self.log(f"Removed {removed} very short questions", "INFO")
        
        # Clean and normalize difficulty labels
        data['difficulty'] = data['difficulty'].str.lower().str.strip()
        data['difficulty'] = data['difficulty'].replace({
            'hard': 'tough',
            'difficult': 'tough',
            'medium': 'moderate',
            'med': 'moderate',
            'avg': 'moderate',
            'simple': 'easy',
            'basic': 'easy'
        })
        
        # Combine text fields with enhanced weighting
        data['combined_text'] = (
            data['question'].astype(str) + ' ' +
            data['option1'].astype(str) + ' ' +
            data['option2'].astype(str) + ' ' +
            data['option3'].astype(str) + ' ' +
            data['option4'].astype(str)
        )
        
        # Add text-based features for better classification
        data['question_length'] = data['question'].str.len()
        data['total_text_length'] = data['combined_text'].str.len()
        data['avg_word_length'] = data['question'].str.split().str.len()
        
        # Extract subject hints from question text (if available)
        subjects = ['physics', 'chemistry', 'biology', 'botany', 'zoology']
        for subject in subjects:
            data[f'contains_{subject}'] = data['question'].str.lower().str.contains(subject).astype(int)
        
        self.log(f"✅ Text preprocessing completed. Final dataset: {len(data)} questions", "SUCCESS")
        return data

    def train_enhanced_difficulty_classifier(self, csv_file):
        """Enhanced difficulty classifier with advanced techniques"""
        self.log("🤖 Training Enhanced Difficulty Classifier", "PROGRESS")
        
        try:
            data = pd.read_csv(csv_file)
        except Exception as e:
            self.log(f"Error reading {csv_file}: {e}", "ERROR")
            return False

        # Validate data quality
        required_columns = ['question', 'option1', 'option2', 'option3', 'option4', 'difficulty']
        is_valid, issues = self.validate_data_quality(data, required_columns, "question dataset")
        
        if not is_valid:
            self.log("Data quality issues prevent training", "ERROR")
            return False

        # Enhanced preprocessing
        data = self.enhanced_text_preprocessing(data)
        
        # Filter valid difficulties
        valid_difficulties = ['easy', 'moderate', 'tough']
        data = data[data['difficulty'].isin(valid_difficulties)].copy()
        
        if len(data) < 50:
            self.log(f"Insufficient data after cleaning: {len(data)} rows", "ERROR")
            return False

        # Check class distribution
        class_dist = data['difficulty'].value_counts()
        self.log(f"Class distribution: {class_dist.to_dict()}")
        
        # Handle severe class imbalance
        min_class_size = class_dist.min()
        if min_class_size < 10:
            self.log("Severe class imbalance detected. Applying SMOTE-like balancing", "WARNING")
            # Simple oversampling for minority classes
            balanced_data = []
            target_size = max(20, min_class_size * 2)
            
            for difficulty in valid_difficulties:
                class_data = data[data['difficulty'] == difficulty]
                if len(class_data) < target_size:
                    # Oversample by repeating with slight text variations
                    additional_needed = target_size - len(class_data)
                    repeated_data = class_data.sample(n=additional_needed, replace=True, random_state=self.random_state)
                    class_data = pd.concat([class_data, repeated_data])
                balanced_data.append(class_data)
            
            data = pd.concat(balanced_data, ignore_index=True)
            self.log(f"Balanced dataset size: {len(data)}")

        # Prepare features and targets
        X = data['combined_text']
        y = data['difficulty']
        
        # Enhanced train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, 
            stratify=y, shuffle=True
        )

        # Enhanced TF-IDF pipeline with multiple configurations
        vectorizer_configs = [
            {
                'max_features': 2000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8,
                'sublinear_tf': True
            },
            {
                'max_features': 3000,
                'ngram_range': (1, 3),
                'min_df': 3,
                'max_df': 0.9,
                'sublinear_tf': True
            },
            {
                'max_features': 1500,
                'ngram_range': (1, 2),
                'min_df': 1,
                'max_df': 0.95,
                'sublinear_tf': False
            }
        ]
        
        best_score = 0
        best_model = None
        
        for i, vec_config in enumerate(vectorizer_configs):
            self.log(f"Testing vectorizer configuration {i+1}/{len(vectorizer_configs)}", "PROGRESS")
            
            # Create pipeline with current configuration
            pipeline = make_pipeline(
                TfidfVectorizer(stop_words='english', **vec_config),
                LinearSVC(class_weight='balanced', random_state=self.random_state, max_iter=2000)
            )
            
            # Enhanced hyperparameter tuning
            param_grid = {
                'linearsvc__C': [0.1, 0.5, 1.0, 2.0, 5.0],
                'linearsvc__loss': ['hinge', 'squared_hinge']
            }
            
            # Use cross-validation for more robust evaluation
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=5, scoring='f1_macro',
                n_jobs=-1, verbose=0
            )
            
            try:
                grid_search.fit(X_train, y_train)
                score = grid_search.best_score_
                
                if score > best_score:
                    best_score = score
                    best_model = grid_search.best_estimator_
                    
                self.log(f"Configuration {i+1} - F1 Score: {score:.3f}")
                
            except Exception as e:
                self.log(f"Configuration {i+1} failed: {e}", "WARNING")
                continue

        if best_model is None:
            self.log("All classifier configurations failed", "ERROR")
            return False

        # Final evaluation
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        
        # Detailed evaluation report
        self.log("📊 Enhanced Classifier Evaluation:", "SUCCESS")
        self.log(f"Best CV F1 Score: {best_score:.3f}")
        self.log(f"Test Accuracy: {test_accuracy:.3f}")
        self.log(f"Test F1 Score: {test_f1:.3f}")
        
        # Per-class performance
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=valid_difficulties)
        
        self.log("\nPer-class Performance:")
        for i, diff in enumerate(valid_difficulties):
            self.log(f"  {diff.capitalize():>10}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

        # Save the enhanced model
        model_path = 'neet_question_classifier.pkl'
        dump(best_model, model_path)
        
        # Save training metadata
        metadata = {
            'model_type': 'Enhanced LinearSVC with TF-IDF',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'cv_f1_score': best_score,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'class_distribution': class_dist.to_dict(),
            'features_used': 'combined_text',
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.models_trained['difficulty_classifier'] = metadata
        
        self.log(f"✅ Enhanced difficulty classifier saved as '{model_path}'", "SUCCESS")
        return True

    def advanced_feature_engineering(self, stats):
        """Advanced feature engineering with domain expertise"""
        self.log("🔧 Applying advanced feature engineering...", "PROGRESS")
        
        # Ensure required columns exist with proper defaults
        required_cols = {
            'easy_questions': 0, 'moderate_questions': 0, 'tough_questions': 0,
            'total_attendees': 1000000, 'year': 2020, 'general_mbbs_cutoff': 500
        }
        
        for col, default_val in required_cols.items():
            if col not in stats.columns:
                stats[col] = default_val
                
        # Sort by year for time-series features
        stats = stats.sort_values('year', na_position='last').reset_index(drop=True)
        
        # Basic derived features
        stats['total_questions'] = (
            stats['easy_questions'] + stats['moderate_questions'] + stats['tough_questions']
        )
        
        # Handle division by zero
        total_q_safe = stats['total_questions'].replace(0, 1)
        
        # Difficulty ratios with enhanced precision
        stats['easy_ratio'] = stats['easy_questions'] / total_q_safe
        stats['moderate_ratio'] = stats['moderate_questions'] / total_q_safe
        stats['tough_ratio'] = stats['tough_questions'] / total_q_safe
        
        # Advanced difficulty metrics
        stats['avg_difficulty'] = (
            (stats['easy_questions'] * 1 + 
             stats['moderate_questions'] * 2 + 
             stats['tough_questions'] * 3) / total_q_safe
        )
        
        # Difficulty variance (measure of distribution spread)
        stats['difficulty_variance'] = (
            (stats['easy_questions'] * (1 - stats['avg_difficulty'])**2 +
             stats['moderate_questions'] * (2 - stats['avg_difficulty'])**2 +
             stats['tough_questions'] * (3 - stats['avg_difficulty'])**2) / total_q_safe
        )
        
        # Advanced attendee features
        stats['attendees_million'] = stats['total_attendees'] / 1_000_000.0
        stats['attendees_log'] = np.log1p(stats['attendees_million'])
        stats['attendees_log10'] = np.log10(stats['total_attendees'].clip(lower=1))
        stats['attendees_sqrt'] = np.sqrt(stats['total_attendees'])
        stats['attendees_squared'] = stats['total_attendees'] ** 2
        stats['attendees_cubed'] = stats['total_attendees'] ** 3
        
        # Year-based features
        if 'year' in stats.columns:
            current_year = 2025
            stats['years_since_2015'] = stats['year'] - 2015
            stats['years_since_2020'] = stats['year'] - 2020  
            stats['years_from_current'] = current_year - stats['year']
            stats['year_squared'] = stats['year'] ** 2
            stats['year_normalized'] = (stats['year'] - stats['year'].min()) / (stats['year'].max() - stats['year'].min())
        
        # Competition intensity features
        if 'available_seats' in stats.columns:
            stats['attendees_per_seat'] = stats['total_attendees'] / stats['available_seats'].clip(lower=1)
            stats['competition_intensity'] = np.log1p(stats['attendees_per_seat'])
        else:
            # Estimate based on typical NEET seat numbers
            estimated_seats = 100000  # Rough estimate of total NEET seats
            stats['attendees_per_seat'] = stats['total_attendees'] / estimated_seats
            stats['competition_intensity'] = np.log1p(stats['attendees_per_seat'])
        
        # Question density and complexity
        stats['question_density'] = stats['total_questions'] / 180.0  # Normalized to standard NEET
        stats['complexity_score'] = stats['tough_ratio'] * 2 + stats['moderate_ratio'] * 1
        stats['easy_dominance'] = stats['easy_ratio'] - stats['tough_ratio']
        
        # Historical trend features
        stats['prev_cutoff'] = stats['general_mbbs_cutoff'].shift(1)
        stats['cutoff_change'] = stats['general_mbbs_cutoff'] - stats['prev_cutoff']
        stats['cutoff_change_rate'] = stats['cutoff_change'] / stats['prev_cutoff'].clip(lower=1)
        
        # Rolling averages for trend analysis
        if len(stats) >= 3:
            stats['cutoff_ma3'] = stats['general_mbbs_cutoff'].rolling(window=3, min_periods=1).mean()
            stats['attendees_ma3'] = stats['total_attendees'].rolling(window=3, min_periods=1).mean()
        
        # Fill NaN values with intelligent defaults
        stats['prev_cutoff'] = stats['prev_cutoff'].fillna(stats['general_mbbs_cutoff'].mean())
        stats['cutoff_change'] = stats['cutoff_change'].fillna(0)
        stats['cutoff_change_rate'] = stats['cutoff_change_rate'].fillna(0)
        
        # Policy/external factors
        if 'policy_change' not in stats.columns:
            stats['policy_change'] = 0
        
        # COVID impact (2020-2022)
        stats['covid_impact'] = ((stats['year'] >= 2020) & (stats['year'] <= 2022)).astype(int)
        
        # Advanced interaction features
        stats['difficulty_attendees'] = stats['avg_difficulty'] * stats['attendees_million']
        stats['competition_difficulty'] = stats['competition_intensity'] * stats['avg_difficulty']
        stats['year_difficulty'] = stats['year_normalized'] * stats['avg_difficulty']
        
        # Percentile-based features for robustness
        for col in ['total_attendees', 'avg_difficulty', 'competition_intensity']:
            if col in stats.columns:
                stats[f'{col}_percentile'] = stats[col].rank(pct=True)
        
        # Fill any remaining NaN values
        numeric_cols = stats.select_dtypes(include=[np.number]).columns
        stats[numeric_cols] = stats[numeric_cols].fillna(stats[numeric_cols].median())
        
        # Replace infinite values
        stats = stats.replace([np.inf, -np.inf], np.nan)
        stats = stats.fillna(0)
        
        self.log(f"✅ Feature engineering complete. Created {len(stats.columns)} features", "SUCCESS")
        return stats

    def prepare_enhanced_merged_stats(self, stats_file, questions_file):
        """Enhanced data merging with comprehensive validation"""
        self.log("📊 Preparing enhanced merged statistics...", "PROGRESS")
        
        try:
            # Load and validate question data
            questions = pd.read_csv(questions_file)
            required_q_cols = ['year', 'difficulty']
            
            is_valid, issues = self.validate_data_quality(questions, required_q_cols, "questions dataset")
            if not is_valid:
                self.log("Using stats file only due to question data issues", "WARNING")
                return pd.read_csv(stats_file)
            
            # Enhanced aggregation with error handling
            agg = questions.groupby('year')['difficulty'].value_counts().unstack(fill_value=0).reset_index()
            
            # Ensure all difficulty columns exist
            for col in ['easy', 'moderate', 'tough']:
                if col not in agg.columns:
                    agg[col] = 0
                    
            agg['total_questions'] = agg[['easy', 'moderate', 'tough']].sum(axis=1)
            agg = agg.rename(columns={
                'easy': 'easy_questions', 
                'moderate': 'moderate_questions', 
                'tough': 'tough_questions'
            })
            
            # Load exam statistics
            stats = pd.read_csv(stats_file)
            
            # Enhanced merging strategy
            merged = pd.merge(stats, agg, on='year', how='outer', suffixes=('_stats', '_agg'))
            
            # Intelligent column reconciliation
            def reconcile_columns(base_name):
                stats_col = f"{base_name}_stats"
                agg_col = f"{base_name}_agg"
                
                if stats_col in merged.columns and agg_col in merged.columns:
                    # Use stats data preferentially, fill with aggregated data
                    result = merged[stats_col].fillna(merged[agg_col])
                    # If both exist, use average for conflicting values
                    both_exist = merged[stats_col].notna() & merged[agg_col].notna()
                    conflicting = both_exist & (abs(merged[stats_col] - merged[agg_col]) > 5)
                    if conflicting.any():
                        result.loc[conflicting] = (merged.loc[conflicting, stats_col] + 
                                                 merged.loc[conflicting, agg_col]) / 2
                elif stats_col in merged.columns:
                    result = merged[stats_col]
                elif agg_col in merged.columns:
                    result = merged[agg_col]
                else:
                    result = pd.Series(0, index=merged.index)
                    
                return result
            
            # Apply reconciliation
            for base in ['easy_questions', 'moderate_questions', 'tough_questions']:
                merged[base] = reconcile_columns(base)
            
            # Clean up temporary columns
            drop_cols = [c for c in merged.columns if c.endswith(('_stats', '_agg'))]
            merged = merged.drop(columns=drop_cols)
            
            # Data quality checks on merged data
            if merged.empty:
                self.log("Merged dataset is empty, using stats only", "WARNING")
                return pd.read_csv(stats_file)
            
            # Report merging results
            stats_years = set(stats['year']) if 'year' in stats.columns else set()
            agg_years = set(agg['year']) if 'year' in agg.columns else set()
            
            if stats_years - agg_years:
                self.log(f"Years in stats but not in questions: {sorted(stats_years - agg_years)}", "INFO")
            if agg_years - stats_years:
                self.log(f"Years in questions but not in stats: {sorted(agg_years - stats_years)}", "INFO")
                
            self.log(f"✅ Successfully merged data for {len(merged)} years", "SUCCESS")
            return merged
            
        except Exception as e:
            self.log(f"Error in data merging: {e}, using stats file only", "ERROR")
            return pd.read_csv(stats_file)

    def train_advanced_cutoff_predictor(self, stats_file, questions_file):
        """Advanced cutoff predictor with enhanced algorithms and validation"""
        self.log("🎯 Training Advanced Cutoff Predictor", "PROGRESS")
        
        # Prepare enhanced dataset
        merged = self.prepare_enhanced_merged_stats(stats_file, questions_file)
        
        if len(merged) < 6:
            self.log("Insufficient data for advanced modeling, using simplified approach", "WARNING")
            merged = pd.read_csv(stats_file)
        
        # Advanced feature engineering
        merged = self.advanced_feature_engineering(merged)
        merged = merged.sort_values('year', na_position='last').reset_index(drop=True)
        
        # Enhanced feature selection
        base_features = [
            'year', 'easy_questions', 'moderate_questions', 'tough_questions',
            'total_attendees', 'total_questions', 'easy_ratio', 'moderate_ratio', 
            'tough_ratio', 'avg_difficulty', 'difficulty_variance', 'question_density',
            'attendees_million', 'attendees_log', 'attendees_log10', 'attendees_sqrt',
            'years_since_2015', 'years_since_2020', 'attendees_per_seat',
            'competition_intensity', 'complexity_score', 'easy_dominance',
            'prev_cutoff', 'cutoff_change', 'covid_impact', 'policy_change',
            'difficulty_attendees', 'competition_difficulty', 'year_difficulty'
        ]
        
        # Advanced features (if available)
        advanced_features = [
            'attendees_squared', 'year_squared', 'attendees_cubed',
            'cutoff_ma3', 'attendees_ma3', 'total_attendees_percentile',
            'avg_difficulty_percentile', 'competition_intensity_percentile'
        ]
        
        # Select available features
        available_features = [f for f in base_features + advanced_features if f in merged.columns]
        self.log(f"Using {len(available_features)} features for training")
        
        # Prepare target variable with realistic bounds
        y = merged['general_mbbs_cutoff'].clip(250, 720)
        X = merged[available_features]
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Enhanced train-test split (time-aware)
        test_size = max(2, min(4, len(X) // 4))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
        self.log(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Advanced model selection with multiple algorithms
        models_config = [
            ('ridge', Ridge(random_state=self.random_state), {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
            }),
            ('lasso', Lasso(random_state=self.random_state, max_iter=2000), {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            }),
            ('elastic', ElasticNet(random_state=self.random_state, max_iter=2000), {
                'alpha': [0.1, 0.5, 1.0, 2.0], 'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }),
            ('gbr', GradientBoostingRegressor(random_state=self.random_state), {
                'n_estimators': [100, 150, 200], 
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 4, 5], 'min_samples_split': [2, 5]
            }),
            ('rf', RandomForestRegressor(random_state=self.random_state), {
                'n_estimators': [100, 150, 200], 'max_depth': [5, 7, 10],
                'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]
            }),
            ('extra', ExtraTreesRegressor(random_state=self.random_state), {
                'n_estimators': [100, 150], 'max_depth': [5, 7],
                'min_samples_split': [2, 5]
            })
        ]
        
        # Enhanced cross-validation strategy
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) - 2)))
        
        best_models = {}
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate each model
        for name, model, param_grid in models_config:
            self.log(f"Training {name.upper()} model...", "PROGRESS")
            
            try:
                # Hyperparameter optimization
                grid_search = GridSearchCV(
                    model, param_grid, cv=tscv, 
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train_scaled, y_train)
                
                # Evaluate on test set
                y_pred = grid_search.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                best_models[name] = {
                    'model': grid_search.best_estimator_,
                    'cv_score': -grid_search.best_score_,
                    'test_mae': mae,
                    'test_r2': r2,
                    'params': grid_search.best_params_
                }
                
                self.log(f"{name.upper()}: CV MAE={-grid_search.best_score_:.2f}, Test MAE={mae:.2f}, R²={r2:.3f}")
                
            except Exception as e:
                self.log(f"Failed to train {name}: {e}", "WARNING")
                continue
        
        if not best_models:
            self.log("All models failed to train", "ERROR")
            return False
        
        # Select best individual model
        best_name = min(best_models.keys(), key=lambda x: best_models[x]['test_mae'])
        best_individual = best_models[best_name]
        
        self.log(f"🏆 Best individual model: {best_name.upper()}", "SUCCESS")
        
        # Retrain best model on full dataset
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X)
        best_individual['model'].fit(X_full_scaled, y)
        
        # Save individual model
        individual_model_data = (best_individual['model'], scaler_full, available_features)
        dump(individual_model_data, 'neet_cutoff_predictor.pkl')
        
        # Training metadata
        metadata = {
            'model_type': f'Enhanced {best_name.upper()}',
            'best_params': best_individual['params'],
            'cv_mae': best_individual['cv_score'],
            'test_mae': best_individual['test_mae'],
            'test_r2': best_individual['test_r2'],
            'n_features': len(available_features),
            'training_samples': len(X_train),
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.models_trained['cutoff_predictor'] = metadata
        
        self.log("✅ Advanced cutoff predictor saved as 'neet_cutoff_predictor.pkl'", "SUCCESS")
        return True

    def train_supreme_ensemble_model(self, stats_file, questions_file):
        """Supreme ensemble model with advanced techniques and meta-learning"""
        self.log("🚀 Training Supreme Ensemble Model", "PROGRESS")
        
        # Prepare enhanced dataset  
        merged = self.prepare_enhanced_merged_stats(stats_file, questions_file)
        
        if len(merged) < 6:
            self.log("Insufficient data for ensemble, skipping", "WARNING")
            return False
        
        # Advanced feature engineering
        merged = self.advanced_feature_engineering(merged)
        merged = merged.sort_values('year', na_position='last').reset_index(drop=True)
        
        # Comprehensive feature set for ensemble
        all_features = [
            'year', 'easy_questions', 'moderate_questions', 'tough_questions',
            'total_attendees', 'total_questions', 'easy_ratio', 'moderate_ratio',
            'tough_ratio', 'avg_difficulty', 'difficulty_variance', 'question_density',
            'attendees_million', 'attendees_log', 'attendees_log10', 'years_since_2015',
            'years_since_2020', 'attendees_per_seat', 'competition_intensity',
            'complexity_score', 'easy_dominance', 'prev_cutoff', 'cutoff_change',
            'covid_impact', 'policy_change', 'difficulty_attendees',
            'competition_difficulty', 'year_difficulty', 'attendees_squared'
        ]
        
        # Select available features
        feature_cols = [f for f in all_features if f in merged.columns]
        self.log(f"Ensemble using {len(feature_cols)} features")
        
        # Prepare data
        y = merged['general_mbbs_cutoff'].clip(250, 720)
        X = merged[feature_cols].fillna(merged[feature_cols].median())
        
        # Enhanced time-series split
        test_size = max(2, min(3, len(X) // 4))
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
        # Multiple scaling strategies for robustness
        scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler(),
            'minmax': MinMaxScaler()
        }
        
        scaled_data = {}
        for scaler_name, scaler in scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            scaled_data[scaler_name] = (X_train_scaled, X_test_scaled, scaler)
        
        # Supreme ensemble with diverse algorithms
        ensemble_models = {
            'ridge_std': Ridge(alpha=1.0, random_state=self.random_state),
            'ridge_robust': Ridge(alpha=2.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.5, random_state=self.random_state, max_iter=2000),
            'elastic': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            'gbr_conservative': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.08, max_depth=3,
                min_samples_split=3, random_state=self.random_state
            ),
            'gbr_aggressive': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.12, max_depth=4,
                min_samples_split=2, random_state=self.random_state
            ),
            'rf_standard': RandomForestRegressor(
                n_estimators=150, max_depth=6, min_samples_split=3,
                random_state=self.random_state
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=5, min_samples_split=4,
                random_state=self.random_state
            ),
            'ada_boost': AdaBoostRegressor(
                n_estimators=100, learning_rate=0.8, random_state=self.random_state
            ),
            'bayesian_ridge': BayesianRidge()
        }
        
        # Train models with different scalers
        trained_ensemble = {}
        model_performances = []
        
        for model_name, model in ensemble_models.items():
            # Select best scaler for each model type
            if 'ridge' in model_name or 'lasso' in model_name or 'elastic' in model_name:
                scaler_type = 'standard'
            elif 'gbr' in model_name or 'rf' in model_name:
                scaler_type = 'robust' 
            else:
                scaler_type = 'minmax'
                
            X_train_scaled, X_test_scaled, scaler = scaled_data[scaler_type]
            
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=TimeSeriesSplit(n_splits=3),
                    scoring='neg_mean_absolute_error'
                )
                cv_mae = -cv_scores.mean()
                
                trained_ensemble[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'scaler_type': scaler_type,
                    'test_mae': mae,
                    'test_r2': r2,
                    'cv_mae': cv_mae,
                    'weight': 1.0 / max(mae, 0.1)  # Performance-based weighting
                }
                
                model_performances.append((model_name, mae, r2))
                self.log(f"{model_name:>15}: Test MAE={mae:.2f}, R²={r2:.3f}, CV MAE={cv_mae:.2f}")
                
            except Exception as e:
                self.log(f"Failed to train {model_name}: {e}", "WARNING")
                continue
        
        if not trained_ensemble:
            self.log("No ensemble models trained successfully", "ERROR")
            return False
        
        # Calculate optimal weights using validation performance
        total_weight = sum(model_data['weight'] for model_data in trained_ensemble.values())
        for model_data in trained_ensemble.values():
            model_data['weight'] /= total_weight
        
        # Create final ensemble prediction function
        def ensemble_predict(X_input):
            predictions = []
            weights = []
            
            for model_name, model_data in trained_ensemble.items():
                try:
                    scaler = model_data['scaler']  
                    model = model_data['model']
                    weight = model_data['weight']
                    
                    X_scaled = scaler.transform(X_input)
                    pred = model.predict(X_scaled)
                    
                    predictions.append(pred)
                    weights.append(weight)
                except:
                    continue
            
            if not predictions:
                return np.full(len(X_input), 500)  # Safe default
                
            # Weighted ensemble prediction
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            
            # Apply realistic bounds
            ensemble_pred = np.clip(ensemble_pred, 250, 720)
            return ensemble_pred
        
        # Test ensemble performance
        ensemble_pred = ensemble_predict(X_test)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        self.log(f"🏆 ENSEMBLE PERFORMANCE: MAE={ensemble_mae:.2f}, R²={ensemble_r2:.3f}", "SUCCESS")
        
        # Retrain all models on full dataset
        X_full = X.values
        for model_name, model_data in trained_ensemble.items():
            scaler = model_data['scaler']
            model = model_data['model']
            
            # Refit scaler and model on full data
            X_full_scaled = scaler.fit_transform(X_full)
            model.fit(X_full_scaled, y)
        
        # Save ensemble as list of models (for compatibility)
        ensemble_models_list = [model_data['model'] for model_data in trained_ensemble.values()]
        ensemble_scalers_list = [model_data['scaler'] for model_data in trained_ensemble.values()]
        ensemble_weights = [model_data['weight'] for model_data in trained_ensemble.values()]
        
        # Save in multiple formats for maximum compatibility
        ensemble_data_dict = (trained_ensemble, 'primary_scaler_placeholder', feature_cols)
        ensemble_data_list = (ensemble_models_list, ensemble_scalers_list, feature_cols, ensemble_weights)
        
        dump(ensemble_data_dict, 'neet_ensemble_predictor.pkl')
        dump(ensemble_data_list, 'neet_ensemble_predictor_list.pkl') 
        
        # Save metadata
        ensemble_metadata = {
            'model_type': 'Supreme Ensemble',
            'n_models': len(trained_ensemble),
            'models_used': list(trained_ensemble.keys()),
            'test_mae': ensemble_mae,
            'test_r2': ensemble_r2,
            'n_features': len(feature_cols),
            'best_individual_mae': min(perf[1] for perf in model_performances),
            'ensemble_improvement': min(perf[1] for perf in model_performances) - ensemble_mae,
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.models_trained['ensemble'] = ensemble_metadata
        
        self.log("✅ Supreme ensemble model saved as 'neet_ensemble_predictor.pkl'", "SUCCESS")
        self.log(f"🎯 Ensemble improvement over best individual: {ensemble_metadata['ensemble_improvement']:.2f} MAE points")
        
        return True

    def generate_training_report(self):
        """Generate comprehensive training report"""
        self.log("📋 Generating comprehensive training report...", "PROGRESS")
        
        report_lines = [
            "="*80,
            f"NEET ANALYZER - MODEL TRAINING REPORT",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "="*80,
            ""
        ]
        
        if not self.models_trained:
            report_lines.append("❌ No models were successfully trained.")
        else:
            report_lines.append(f"✅ Successfully trained {len(self.models_trained)} model(s)")
            report_lines.append("")
            
            for model_name, metadata in self.models_trained.items():
                report_lines.extend([
                    f"📊 {model_name.upper().replace('_', ' ')}:",
                    f"   Model Type: {metadata.get('model_type', 'N/A')}",
                    f"   Training Date: {metadata.get('training_date', 'N/A')}"
                ])
                
                if 'test_mae' in metadata:
                    report_lines.append(f"   Test MAE: {metadata['test_mae']:.2f}")
                if 'test_r2' in metadata:
                    report_lines.append(f"   Test R²: {metadata['test_r2']:.3f}")
                if 'test_accuracy' in metadata:
                    report_lines.append(f"   Test Accuracy: {metadata['test_accuracy']:.3f}")
                if 'cv_mae' in metadata:
                    report_lines.append(f"   CV MAE: {metadata['cv_mae']:.2f}")
                if 'n_features' in metadata:
                    report_lines.append(f"   Features Used: {metadata['n_features']}")
                if 'training_samples' in metadata:
                    report_lines.append(f"   Training Samples: {metadata['training_samples']}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "💡 RECOMMENDATIONS:",
            "-" * 40,
        ])
        
        if 'ensemble' in self.models_trained:
            report_lines.append("✅ Ensemble model trained - use for best accuracy")
        else:
            report_lines.append("⚠️ Consider training ensemble model for better performance")
            
        if 'difficulty_classifier' in self.models_trained:
            classifier_acc = self.models_trained['difficulty_classifier'].get('test_accuracy', 0)
            if classifier_acc >= 0.8:
                report_lines.append("✅ Difficulty classifier performance is excellent")
            elif classifier_acc >= 0.7:
                report_lines.append("⚠️ Difficulty classifier is adequate but could be improved")
            else:
                report_lines.append("❌ Difficulty classifier needs more training data")
        
        report_lines.extend([
            "",
            "🔄 To improve model performance:",
            "   1. Collect more diverse training data",
            "   2. Add more years to historical statistics", 
            "   3. Include domain expert annotations",
            "   4. Regular model retraining with new data",
            "",
            "="*80
        ])
        
        # Save report to file
        report_content = "\n".join(report_lines)
        report_path = f"reports/training_report_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.log(f"📋 Training report saved to: {report_path}", "SUCCESS")
        except Exception as e:
            self.log(f"Failed to save report: {e}", "WARNING")
        
        # Print report to console
        print("\n" + report_content)
        
        return report_content

def main():
    """Main training pipeline"""
    print("🚀 NEET ANALYZER - Enhanced Model Training Suite v3.0")
    print("="*70)
    
    # Initialize trainer
    trainer = EnhancedNEETModelTrainer(random_state=42, verbose=True)
    
    # Check for required files
    required_files = [
        'neet_sample_data_cleaned.csv',
        'stats.csv'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        trainer.log(f"❌ Missing required files: {missing_files}", "ERROR")
        trainer.log("Please ensure all training data files are present", "ERROR")
        return
    
    trainer.log("🔍 All required files found", "SUCCESS")
    
    # Training pipeline
    training_start = time.time()
    
    try:
        # Phase 1: Train Enhanced Difficulty Classifier
        trainer.log("PHASE 1: Enhanced Difficulty Classifier", "PROGRESS")
        success_classifier = trainer.train_enhanced_difficulty_classifier('neet_sample_data_cleaned.csv')
        
        # Phase 2: Train Advanced Cutoff Predictor
        trainer.log("PHASE 2: Advanced Cutoff Predictor", "PROGRESS") 
        success_cutoff = trainer.train_advanced_cutoff_predictor('stats.csv', 'neet_sample_data_cleaned.csv')
        
        # Phase 3: Train Supreme Ensemble Model
        trainer.log("PHASE 3: Supreme Ensemble Model", "PROGRESS")
        success_ensemble = trainer.train_supreme_ensemble_model('stats.csv', 'neet_sample_data_cleaned.csv')
        
        # Training completion
        training_time = time.time() - training_start
        trainer.log(f"⏱️ Total training time: {training_time:.1f} seconds", "SUCCESS")
        
        # Generate comprehensive report
        trainer.generate_training_report()
        
        # Final status
        models_created = sum([success_classifier, success_cutoff, success_ensemble])
        trainer.log(f"🏆 Training complete! {models_created}/3 models created successfully", "SUCCESS")
        
        if models_created == 3:
            trainer.log("🌟 All models trained successfully! Your NEET analyzer is ready for maximum accuracy", "SUCCESS")
        elif models_created >= 1:
            trainer.log("⚠️ Some models trained successfully. The analyzer will work with fallbacks", "WARNING")
        else:
            trainer.log("❌ No models trained successfully. Check your data and try again", "ERROR")
            
    except KeyboardInterrupt:
        trainer.log("Training interrupted by user", "WARNING")
    except Exception as e:
        trainer.log(f"Training failed with error: {e}", "ERROR")
        raise

if __name__ == "__main__":
    main()
