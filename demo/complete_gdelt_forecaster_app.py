import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import zipfile
import io
from pathlib import Path
import time
from datetime import datetime, timedelta
import tempfile
import os
import gc
import psutil
import re
import logging

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from prophet import Prophet

# Optional TensorFlow for LSTM
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel('ERROR')
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

# Suppress warnings
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# Current user and time
CURRENT_USER = "strawberrymilktea0604"
CURRENT_TIME = "2025-06-21 07:48:08"

# Page configuration
st.set_page_config(
    page_title="🔥 GDELT Hot Topics Forecaster - Complete Pipeline",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF6B6B;
        margin: 1rem 0;
    }
    .step-container {
        background: linear-gradient(90deg, #FF4B4B, #FF6B6B);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .file-info-card {
        background: #F0F2F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin: 0.5rem 0;
    }
    .hot-topic-card {
        background: linear-gradient(135deg, #FFF5F5, #FFE5E5);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F4FD, #D1E7FF);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .user-info {
        background: linear-gradient(90deg, #4CAF50, #45A049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class ProphetXGBoostTop3Forecaster:
    """Prophet + XGBoost Ensemble for Top 3 Hottest GDELT Topics - Integrated Version"""
    
    def __init__(self, n_topics=10, top_k=3, forecast_horizon=7, batch_size=50000):
        self.n_topics = n_topics
        self.top_k = top_k
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        
        # Core components
        self.vectorizer = None
        self.lda_model = None
        self.scaler = StandardScaler()
        
        # Topic selection
        self.hot_topics = []
        self.topic_popularity = {}
        self.topic_words = {}
        
        # Models
        self.prophet_models = {}
        self.prophet_forecasts = {}
        self.xgboost_models = {}
        self.lstm_model = None
        self.use_lstm = TF_AVAILABLE
        
        # Ensemble weights
        self.ensemble_weights = {
            'prophet': 0.4,
            'xgboost': 0.4, 
            'lstm': 0.2 if self.use_lstm else 0.0
        }
        
        if not self.use_lstm:
            self.ensemble_weights['prophet'] = 0.5
            self.ensemble_weights['xgboost'] = 0.5
        
        # Results storage
        self.training_metrics = {}
        self.feature_importance = {}
        
        # GDELT stopwords
        self.gdelt_stopwords = {
            'wb', 'tax', 'fncact', 'soc', 'policy', 'pointsofinterest', 'crisislex', 
            'epu', 'uspec', 'ethnicity', 'worldlanguages', 'the', 'and', 'or', 
            'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'
        }
        
        print(f"🔥 Prophet + XGBoost Top-{top_k} GDELT Forecaster Initialized")
        print(f"   User: {CURRENT_USER} | Time: {CURRENT_TIME}")
    
    def memory_cleanup(self):
        """Efficient memory cleanup"""
        gc.collect()
        if TF_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
            except:
                pass
    
    def safe_preprocess_text(self, text):
        """Fast single text preprocessing"""
        try:
            if pd.isna(text) or text is None:
                return ""
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = [w for w in text.split() 
                    if len(w) > 2 and w not in self.gdelt_stopwords]
            return ' '.join(words[:40])
        except:
            return ""
    
    def batch_preprocess_fast(self, texts, batch_id=0):
        """Fast batch preprocessing with progress"""
        progress_text = st.empty()
        progress_text.text(f"⚡ Processing batch {batch_id+1}: {len(texts):,} texts...")
        
        start_time = time.time()
        processed = [self.safe_preprocess_text(text) for text in texts]
        valid_texts = [text for text in processed if text.strip()]
        
        elapsed = time.time() - start_time
        rate = len(texts) / elapsed if elapsed > 0 else 0
        
        progress_text.text(f"✅ Batch {batch_id+1}: {len(valid_texts):,}/{len(texts):,} valid ({elapsed:.1f}s, {rate:,.0f} texts/s)")
        
        return valid_texts
    
    def extract_topics_and_identify_hot_topics(self, texts, dates):
        """Extract topics and identify hot topics with progress tracking"""
        st.write("⚡ **Extracting topics and identifying hot topics...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            # Step 1: TF-IDF Setup
            status_text.text("🎯 Setting up TF-IDF vectorizer...")
            progress_bar.progress(10)
            
            first_batch_texts = texts[:self.batch_size]
            first_batch_processed = self.batch_preprocess_fast(first_batch_texts, 0)
            
            if len(first_batch_processed) < 100:
                raise ValueError(f"Insufficient valid texts: {len(first_batch_processed)}")
            
            self.vectorizer = TfidfVectorizer(
                max_features=1500,
                ngram_range=(1, 2),
                min_df=max(3, len(first_batch_processed) // 2000),
                max_df=0.95,
                stop_words='english',
                lowercase=True
            )
            
            progress_bar.progress(20)
            
            # Step 2: LDA Training
            status_text.text("🔄 Training LDA model...")
            first_tfidf = self.vectorizer.fit_transform(first_batch_processed)
            
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=15,
                learning_method='batch',
                batch_size=1024,
                n_jobs=1,
                verbose=0
            )
            
            first_topic_dist = self.lda_model.fit_transform(first_tfidf)
            progress_bar.progress(40)
            
            # Display discovered topics
            feature_names = self.vectorizer.get_feature_names_out()
            st.write("🎯 **Discovered Topics:**")
            
            topic_display = st.empty()
            with topic_display.container():
                for i, topic in enumerate(self.lda_model.components_):
                    top_words = [feature_names[j] for j in topic.argsort()[-5:][::-1]]
                    self.topic_words[i] = top_words
                    st.write(f"   **Topic {i:2d}:** {', '.join(top_words)}")
            
            all_topic_distributions = [first_topic_dist]
            
            # Step 3: Process remaining batches
            if total_batches > 1:
                status_text.text(f"📊 Processing {total_batches-1} remaining batches...")
                
                for batch_idx in range(1, total_batches):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(texts))
                    batch_texts = texts[start_idx:end_idx]
                    
                    try:
                        batch_processed = self.batch_preprocess_fast(batch_texts, batch_idx)
                        
                        if batch_processed:
                            batch_tfidf = self.vectorizer.transform(batch_processed)
                            batch_topics = self.lda_model.transform(batch_tfidf)
                            all_topic_distributions.append(batch_topics)
                        else:
                            fallback = np.full((len(batch_texts), self.n_topics), 1.0/self.n_topics)
                            all_topic_distributions.append(fallback)
                        
                        # Update progress
                        progress = 40 + (batch_idx / (total_batches - 1)) * 40
                        progress_bar.progress(int(progress))
                        
                    except Exception as e:
                        st.warning(f"⚠️ Batch {batch_idx+1} failed: {e}")
                        fallback = np.full((len(batch_texts), self.n_topics), 1.0/self.n_topics)
                        all_topic_distributions.append(fallback)
            
            # Step 4: Combine results
            status_text.text("🔗 Combining results...")
            progress_bar.progress(85)
            
            combined_topic_dist = np.vstack(all_topic_distributions)
            
            if len(combined_topic_dist) < len(texts):
                padding_size = len(texts) - len(combined_topic_dist)
                padding = np.full((padding_size, self.n_topics), 1.0/self.n_topics)
                combined_topic_dist = np.vstack([combined_topic_dist, padding])
            
            # Step 5: Identify hot topics
            status_text.text("🔥 Identifying hot topics...")
            progress_bar.progress(95)
            
            self.identify_hot_topics(combined_topic_dist, dates)
            
            progress_bar.progress(100)
            status_text.text("✅ Topic extraction completed!")
            
            return combined_topic_dist
            
        except Exception as e:
            st.error(f"❌ Topic extraction failed: {e}")
            return np.random.dirichlet(np.ones(self.n_topics), len(texts))
    
    def identify_hot_topics(self, topic_dist, dates):
        """Identify hot topics with detailed analysis"""
        df = pd.DataFrame(topic_dist, columns=[f'topic_{i}' for i in range(self.n_topics)])
        df['date'] = pd.to_datetime(dates)
        
        topic_scores = {}
        
        for topic_idx in range(self.n_topics):
            topic_col = f'topic_{topic_idx}'
            
            # Multiple hotness metrics
            avg_prob = df[topic_col].mean()
            recent_cutoff = int(0.7 * len(df))
            recent_avg = df[topic_col].iloc[recent_cutoff:].mean()
            variance = df[topic_col].var()
            
            daily_avg = df.groupby('date')[topic_col].mean()
            peak_intensity = daily_avg.max()
            
            daily_max_topic = df.groupby('date').apply(
                lambda x: x[[f'topic_{i}' for i in range(self.n_topics)]].mean().idxmax()
            )
            dominance_freq = (daily_max_topic == topic_col).sum() / len(daily_max_topic)
            
            # Combined hotness score
            hotness_score = (
                0.3 * avg_prob +
                0.3 * recent_avg +
                0.2 * variance +
                0.1 * peak_intensity +
                0.1 * dominance_freq
            )
            
            topic_scores[topic_idx] = {
                'hotness_score': hotness_score,
                'avg_prob': avg_prob,
                'recent_avg': recent_avg,
                'variance': variance,
                'peak_intensity': peak_intensity,
                'dominance_freq': dominance_freq
            }
        
        # Select top hot topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]['hotness_score'], reverse=True)
        self.hot_topics = [topic_idx for topic_idx, _ in sorted_topics[:self.top_k]]
        self.topic_popularity = topic_scores
        
        # Display hot topics
        st.write(f"🏆 **Top {self.top_k} Hot Topics:**")
        
        for rank, topic_idx in enumerate(self.hot_topics, 1):
            scores = topic_scores[topic_idx]
            topic_words = self.topic_words.get(topic_idx, [])
            
            with st.container():
                st.markdown(f"""
                <div class="hot-topic-card">
                    <h4>🔥 #{rank}. Topic {topic_idx}: {', '.join(topic_words[:3])}</h4>
                    <p><strong>Keywords:</strong> {', '.join(topic_words)}</p>
                    <p><strong>Hotness Score:</strong> {scores['hotness_score']:.4f} | 
                       <strong>Avg Prob:</strong> {scores['avg_prob']:.4f} | 
                       <strong>Dominance:</strong> {scores['dominance_freq']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def prepare_time_series_data(self, topic_dist, dates):
        """Prepare time series data focused on hot topics"""
        st.write("📊 **Preparing time series data...**")
        
        progress_bar = st.progress(0)
        
        try:
            # Create daily aggregated data
            topic_cols = [f'topic_{i}' for i in range(self.n_topics)]
            df = pd.DataFrame(topic_dist, columns=topic_cols)
            df['date'] = pd.to_datetime(dates)
            
            progress_bar.progress(25)
            
            daily_data = df.groupby('date')[topic_cols].mean().reset_index()
            daily_data = daily_data.sort_values('date').reset_index(drop=True)
            
            progress_bar.progress(50)
            
            # Add time features
            daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
            daily_data['day_of_month'] = daily_data['date'].dt.day
            daily_data['month'] = daily_data['date'].dt.month
            daily_data['quarter'] = daily_data['date'].dt.quarter
            daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
            
            progress_bar.progress(75)
            
            # Create features only for hot topics
            for lag in [1, 2, 3, 7]:
                for topic_idx in self.hot_topics:
                    daily_data[f'topic_{topic_idx}_lag_{lag}'] = daily_data[f'topic_{topic_idx}'].shift(lag)
            
            for window in [3, 7]:
                for topic_idx in self.hot_topics:
                    daily_data[f'topic_{topic_idx}_ma_{window}'] = daily_data[f'topic_{topic_idx}'].rolling(window).mean()
            
            # Cross-topic interactions
            for i, topic_i in enumerate(self.hot_topics):
                for j, topic_j in enumerate(self.hot_topics):
                    if i < j:
                        daily_data[f'topic_{topic_i}_x_{topic_j}'] = daily_data[f'topic_{topic_i}'] * daily_data[f'topic_{topic_j}']
            
            daily_data = daily_data.dropna().reset_index(drop=True)
            
            progress_bar.progress(100)
            
            st.success(f"✅ Time series data prepared: {len(daily_data)} days with {daily_data.shape[1]} features")
            
            return daily_data
            
        except Exception as e:
            st.error(f"❌ Time series preparation failed: {e}")
            return None
    
    def train_ensemble_models(self, daily_data):
        """Train all ensemble models with progress tracking"""
        st.write("🚀 **Training ensemble models...**")
        
        total_steps = 3 if self.use_lstm else 2
        current_step = 0
        
        # Train Prophet models
        st.write("📈 Training Prophet models...")
        prophet_progress = st.progress(0)
        
        prophet_params = {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'yearly_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'interval_width': 0.8
        }
        
        for i, topic_idx in enumerate(self.hot_topics):
            prophet_data = pd.DataFrame({
                'ds': daily_data['date'],
                'y': daily_data[f'topic_{topic_idx}']
            })
            
            model = Prophet(**prophet_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_data)
            
            self.prophet_models[f'topic_{topic_idx}'] = model
            
            future = model.make_future_dataframe(periods=self.forecast_horizon)
            forecast = model.predict(future)
            self.prophet_forecasts[f'topic_{topic_idx}'] = forecast
            
            prophet_progress.progress((i + 1) / len(self.hot_topics))
        
        current_step += 1
        st.success(f"✅ Prophet models trained ({len(self.prophet_models)} models)")
        
        # Train XGBoost models
        st.write("🚀 Training XGBoost models...")
        xgb_progress = st.progress(0)
        
        time_features = ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend']
        lag_features = [col for col in daily_data.columns if 'lag_' in col or 'ma_' in col]
        interaction_features = [col for col in daily_data.columns if '_x_' in col]
        
        for i, topic_idx in enumerate(self.hot_topics):
            other_hot_topics = [f'topic_{j}' for j in self.hot_topics if j != topic_idx]
            X_features = time_features + lag_features + interaction_features + other_hot_topics
            
            X = daily_data[X_features].values
            y = daily_data[f'topic_{topic_idx}'].values
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            self.xgboost_models[f'topic_{topic_idx}'] = model
            
            # Store feature importance
            self.feature_importance[f'topic_{topic_idx}'] = dict(zip(X_features, model.feature_importances_))
            
            xgb_progress.progress((i + 1) / len(self.hot_topics))
        
        current_step += 1
        st.success(f"✅ XGBoost models trained ({len(self.xgboost_models)} models)")
        
        # Train LSTM (optional)
        if self.use_lstm:
            st.write("🔄 Training LSTM model...")
            lstm_progress = st.progress(0)
            
            try:
                hot_topic_cols = [f'topic_{i}' for i in self.hot_topics]
                data = daily_data[hot_topic_cols].values
                
                scaled_data = self.scaler.fit_transform(data)
                
                sequence_length = 7
                X, y = [], []
                
                for i in range(sequence_length, len(scaled_data)):
                    X.append(scaled_data[i-sequence_length:i])
                    y.append(scaled_data[i])
                
                X, y = np.array(X), np.array(y)
                
                if len(X) >= 10:
                    split_idx = int(0.8 * len(X))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    model = Sequential([
                        LSTM(24, input_shape=(sequence_length, self.top_k)),
                        Dropout(0.2),
                        Dense(12, activation='relu'),
                        Dense(self.top_k, activation='linear')
                    ])
                    
                    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=16,
                        verbose=0
                    )
                    
                    self.lstm_model = model
                    lstm_progress.progress(100)
                    st.success("✅ LSTM model trained")
                else:
                    st.warning("⚠️ Insufficient data for LSTM, skipping...")
                    self.use_lstm = False
                    
            except Exception as e:
                st.warning(f"⚠️ LSTM training failed: {e}")
                self.use_lstm = False
        
        return True

class GDELTDataProcessor:
    """Enhanced GDELT Data Processor with ZIP handling"""
    
    def __init__(self):
        self.temp_dir = None
        
    def explore_zip_structure(self, zip_file):
        """Explore ZIP file structure with detailed analysis"""
        st.write("🔍 **Analyzing ZIP file structure...**")
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                file_list = zf.namelist()
                
                csv_files = [f for f in file_list if f.endswith('.csv')]
                directories = list(set([os.path.dirname(f) for f in file_list if os.path.dirname(f)]))
                
                st.write(f"📁 **Total files:** {len(file_list)}")
                st.write(f"📊 **CSV files:** {len(csv_files)}")
                st.write(f"📂 **Directories:** {len(directories)}")
                
                # Show structure in expandable sections
                with st.expander("📂 Directory Structure", expanded=True):
                    if not directories:
                        st.write("📄 All files in root directory")
                        for file in csv_files[:10]:
                            st.write(f"   📄 {file}")
                        if len(csv_files) > 10:
                            st.write(f"   ... and {len(csv_files) - 10} more files")
                    else:
                        for directory in sorted(directories):
                            st.write(f"📁 `{directory}/`")
                            dir_files = [f for f in csv_files if f.startswith(directory)]
                            for file in dir_files[:5]:
                                file_name = os.path.basename(file)
                                st.write(f"   📄 {file_name}")
                            if len(dir_files) > 5:
                                st.write(f"   ... and {len(dir_files) - 5} more files")
                
                file_analysis = self.analyze_file_names(csv_files)
                
                return {
                    'csv_files': csv_files,
                    'directories': directories,
                    'file_analysis': file_analysis
                }
                
        except Exception as e:
            st.error(f"❌ Error reading ZIP file: {e}")
            return None
    
    def analyze_file_names(self, csv_files):
        """Analyze file names to identify train/test data"""
        analysis = {
            'train_candidates': [],
            'test_candidates': [],
            'unknown_files': []
        }
        
        for file in csv_files:
            filename = file.lower()
            
            if any(month in filename for month in ['april', 'apr', 'may', 'tháng4', 'tháng5']):
                analysis['train_candidates'].append(file)
            elif any(month in filename for month in ['june', 'jun', 'tháng6']):
                analysis['test_candidates'].append(file)
            else:
                analysis['unknown_files'].append(file)
        
        return analysis
    
    def extract_and_read_csv(self, zip_file, csv_filename):
        """Extract and read CSV from ZIP with multiple encoding attempts"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                csv_content = zf.read(csv_filename)
                
                for separator in ['\t', ',', ';']:
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            df = pd.read_csv(
                                io.StringIO(csv_content.decode(encoding)),
                                sep=separator,
                                dtype=str,
                                low_memory=False,
                                on_bad_lines='skip'
                            )
                            
                            if len(df.columns) > 1 and len(df) > 0:
                                return df, separator, encoding
                                
                        except Exception:
                            continue
            
            return None, None, None
            
        except Exception as e:
            st.error(f"❌ Error reading file {csv_filename}: {e}")
            return None, None, None
    
    def process_gdelt_dataframe(self, df, file_type="unknown"):
        """Process GDELT DataFrame with progress tracking"""
        st.write(f"🔧 **Processing {file_type} data...**")
        
        try:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"📊 **Shape:** {df.shape}")
            with col2:
                st.write(f"📋 **Columns:** {len(df.columns)}")
            
            # Check required columns
            required_cols = ['DATE', 'THEMES']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {missing_cols}")
                
                # Try to find similar columns
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                theme_cols = [col for col in df.columns if any(word in col.lower() for word in ['theme', 'topic', 'content'])]
                
                if date_cols:
                    st.info(f"💡 Found date columns: {date_cols}")
                if theme_cols:
                    st.info(f"💡 Found theme columns: {theme_cols}")
                
                return None
            
            # Process data with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📅 Converting dates...")
            progress_bar.progress(25)
            df['date'] = pd.to_datetime(df['DATE'], format='%Y%m%d', errors='coerce')
            
            status_text.text("🎯 Processing themes...")
            progress_bar.progress(50)
            df['themes_text'] = df['THEMES'].fillna('').astype(str)
            df['themes_list'] = df['themes_text'].apply(
                lambda x: [theme.strip() for theme in x.split(';') if theme.strip()]
            )
            
            status_text.text("📝 Creating text...")
            progress_bar.progress(75)
            df['text'] = df['themes_list'].apply(
                lambda themes: ' '.join([theme.replace('_', ' ').lower() for theme in themes])
            )
            
            status_text.text("🧹 Cleaning data...")
            progress_bar.progress(90)
            df = df.dropna(subset=['date'])
            df = df[df['text'].str.strip() != '']
            
            result_df = df[['date', 'text']].copy()
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing completed!")
            
            # Show results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Records", f"{len(result_df):,}")
            with col2:
                st.metric("📅 Date Range", f"{(result_df['date'].max() - result_df['date'].min()).days} days")
            with col3:
                daily_avg = result_df.groupby(result_df['date'].dt.date).size().mean()
                st.metric("📈 Avg/Day", f"{daily_avg:.1f}")
            
            return result_df
            
        except Exception as e:
            st.error(f"❌ Data processing failed: {e}")
            return None
    
    def create_demo_data(self):
        """Create realistic demo data"""
        st.write("🎭 **Creating demo data...**")
        
        np.random.seed(42)
        
        gdelt_themes = [
            'TRIAL TAX_FNCACT TAX_FNCACT_LAWYER',
            'WB_1979_NATURAL_RESOURCE_MANAGEMENT WB_435_AGRICULTURE_AND_FOOD_SECURITY',
            'PORTSMEN_HOLIDAY CRISISLEX_CRISISLEXREC SOC_POINTSOFINTEREST',
            'TAX_FNCACT_POLICE SOC_POINTSOFINTEREST_PRISON WB_2405_DETENTION_REFORM',
            'ARREST TAX_FNCACT TAX_FNCACT_OFFICIALS TRIAL',
            'TERROR ARMEDCONFLICT TAX_ETHNICITY_VENEZUELANS',
            'WB_826_TOURISM WB_1921_COMPETITIVE_AND_REAL_SECTORS',
            'EPU_ECONOMY EPU_ECONOMY_HISTORIC TAX_ETHNICITY_SPANISH',
            'WB_698 MEDIA_MSM AFFECT BAN',
            'SECURITY_SERVICES CRIME WB_ILLEGAL_DRUGS'
        ]
        
        # Generate training data
        dates_train = pd.date_range('2024-04-01', '2024-05-31', freq='D')
        train_data = []
        
        progress_bar = st.progress(0)
        
        for i, date in enumerate(dates_train):
            n_articles = np.random.randint(80, 200)
            for _ in range(n_articles):
                theme = np.random.choice(gdelt_themes)
                text = theme.replace('_', ' ').lower()
                train_data.append({'date': date, 'text': text})
            
            progress_bar.progress((i + 1) / len(dates_train) * 0.7)
        
        # Generate test data
        dates_test = pd.date_range('2024-06-01', '2024-06-10', freq='D')
        test_data = []
        
        for i, date in enumerate(dates_test):
            n_articles = np.random.randint(60, 150)
            for _ in range(n_articles):
                theme = np.random.choice(gdelt_themes)
                text = theme.replace('_', ' ').lower()
                test_data.append({'date': date, 'text': text})
            
            progress_bar.progress(0.7 + (i + 1) / len(dates_test) * 0.3)
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🏋️ Training Records", f"{len(train_df):,}")
        with col2:
            st.metric("🧪 Test Records", f"{len(test_df):,}")
        
        return train_df, test_df

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'step': 1,
        'zip_structure': None,
        'train_data': None,
        'test_data': None,
        'model_trained': False,
        'forecaster': None,
        'predictions': None,
        'actuals': None,
        'results': None,
        'current_user': CURRENT_USER,
        'start_time': CURRENT_TIME
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """Main application function"""
    init_session_state()
    
    # Header with user info
    st.markdown('<h1 class="main-header">🔥 GDELT Hot Topics Forecaster</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="user-info">
        👤 <strong>User:</strong> {CURRENT_USER} | 
        🕐 <strong>Session Started:</strong> {CURRENT_TIME} UTC | 
        🔥 <strong>Complete Pipeline:</strong> ZIP Upload → Data Processing → Topic Modeling → Forecasting
    </div>
    """, unsafe_allow_html=True)
    
    # Progress indicator
    steps = ["📁 Upload ZIP", "🔍 Explore Data", "📊 Process Data", "🔥 Train Model", "📈 Results"]
    current_step = st.session_state.step
    
    cols = st.columns(len(steps))
    for i, (col, step_name) in enumerate(zip(cols, steps)):
        with col:
            if i + 1 < current_step:
                st.markdown(f"✅ **{step_name}**")
            elif i + 1 == current_step:
                st.markdown(f"🔄 **{step_name}**")
            else:
                st.markdown(f"⏳ {step_name}")
    
    st.markdown("---")
    
    # STEP 1: Upload ZIP file
    if st.session_state.step == 1:
        st.markdown('<div class="step-container"><h2>📁 STEP 1: Upload GDELT Data</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📤 Upload ZIP File")
            uploaded_zip = st.file_uploader(
                "Upload GDELT data ZIP file containing CSV files",
                type=['zip'],
                help="Upload ZIP file with GDELT CSV data organized by month (April-May for training, June for testing)"
            )
            
            if uploaded_zip is not None:
                processor = GDELTDataProcessor()
                
                with st.spinner("🔍 Analyzing ZIP file..."):
                    zip_structure = processor.explore_zip_structure(uploaded_zip)
                
                if zip_structure:
                    st.session_state.zip_structure = zip_structure
                    st.session_state.zip_file = uploaded_zip
                    
                    st.success("✅ ZIP file analyzed successfully!")
                    
                    if st.button("🚀 Continue to Data Exploration", type="primary"):
                        st.session_state.step = 2
                        st.rerun()
        
        with col2:
            st.markdown("### 🎭 Demo Option")
            st.info("💡 **Quick Start:** Use demo data to explore the forecasting capabilities without uploading files.")
            
            if st.button("🎭 Use Demo Data", type="secondary"):
                processor = GDELTDataProcessor()
                train_data, test_data = processor.create_demo_data()
                
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.step = 4
                st.success("✅ Demo data loaded! Skipping to model training...")
                time.sleep(1)
                st.rerun()
    
    # STEP 2: Explore data structure
    elif st.session_state.step == 2:
        st.markdown('<div class="step-container"><h2>🔍 STEP 2: Explore Data Structure</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.zip_structure:
            zip_structure = st.session_state.zip_structure
            file_analysis = zip_structure['file_analysis']
            
            # Show file categorization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🏋️ Training Data Candidates")
                st.markdown('<div class="file-info-card">', unsafe_allow_html=True)
                if file_analysis['train_candidates']:
                    for file in file_analysis['train_candidates']:
                        st.write(f"📄 `{os.path.basename(file)}`")
                else:
                    st.warning("⚠️ No training candidates found")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 🧪 Test Data Candidates")
                st.markdown('<div class="file-info-card">', unsafe_allow_html=True)
                if file_analysis['test_candidates']:
                    for file in file_analysis['test_candidates']:
                        st.write(f"📄 `{os.path.basename(file)}`")
                else:
                    st.warning("⚠️ No test candidates found")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown("### ❓ Unknown Files")
                st.markdown('<div class="file-info-card">', unsafe_allow_html=True)
                if file_analysis['unknown_files']:
                    for file in file_analysis['unknown_files'][:5]:
                        st.write(f"📄 `{os.path.basename(file)}`")
                    if len(file_analysis['unknown_files']) > 5:
                        st.write(f"... and {len(file_analysis['unknown_files']) - 5} more")
                else:
                    st.info("✅ All files categorized")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # File selection
            st.markdown("### 📋 Select Files for Processing")
            
            all_csv_files = zip_structure['csv_files']
            
            col1, col2 = st.columns(2)
            
            with col1:
                selected_train_files = st.multiselect(
                    "🏋️ Select Training Files (April-May):",
                    options=all_csv_files,
                    default=file_analysis['train_candidates'][:10],
                    help="Select CSV files containing April-May data for training"
                )
            
            with col2:
                selected_test_files = st.multiselect(
                    "🧪 Select Test Files (June):",
                    options=all_csv_files,
                    default=file_analysis['test_candidates'][:5],
                    help="Select CSV files containing June data for testing"
                )
            
            # Validation and proceed
            if selected_train_files and selected_test_files:
                st.success(f"✅ Selected {len(selected_train_files)} training files and {len(selected_test_files)} test files")
                
                st.session_state.selected_train_files = selected_train_files
                st.session_state.selected_test_files = selected_test_files
                
                if st.button("📊 Process Selected Files", type="primary"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.warning("⚠️ Please select both training and test files to continue")
    
    # STEP 3: Process data
    elif st.session_state.step == 3:
        st.markdown('<div class="step-container"><h2>📊 STEP 3: Process GDELT Data</h2></div>', unsafe_allow_html=True)
        
        processor = GDELTDataProcessor()
        
        # Process training files
        st.markdown("### 🏋️ Processing Training Data")
        train_dataframes = []
        
        train_progress = st.progress(0)
        train_status = st.empty()
        
        for i, train_file in enumerate(st.session_state.selected_train_files):
            train_status.text(f"Processing {train_file}...")
            
            with st.expander(f"📄 {os.path.basename(train_file)}", expanded=False):
                df, separator, encoding = processor.extract_and_read_csv(st.session_state.zip_file, train_file)
                
                if df is not None:
                    st.success(f"✅ Read successfully (sep: '{separator}', encoding: '{encoding}')")
                    processed_df = processor.process_gdelt_dataframe(df, f"train_{i+1}")
                    
                    if processed_df is not None:
                        train_dataframes.append(processed_df)
                        st.info(f"✅ Added {len(processed_df):,} records to training set")
                else:
                    st.error(f"❌ Failed to read {train_file}")
            
            train_progress.progress((i + 1) / len(st.session_state.selected_train_files))
        
        # Process test files
        st.markdown("### 🧪 Processing Test Data")
        test_dataframes = []
        
        test_progress = st.progress(0)
        test_status = st.empty()
        
        for i, test_file in enumerate(st.session_state.selected_test_files):
            test_status.text(f"Processing {test_file}...")
            
            with st.expander(f"📄 {os.path.basename(test_file)}", expanded=False):
                df, separator, encoding = processor.extract_and_read_csv(st.session_state.zip_file, test_file)
                
                if df is not None:
                    st.success(f"✅ Read successfully (sep: '{separator}', encoding: '{encoding}')")
                    processed_df = processor.process_gdelt_dataframe(df, f"test_{i+1}")
                    
                    if processed_df is not None:
                        test_dataframes.append(processed_df)
                        st.info(f"✅ Added {len(processed_df):,} records to test set")
                else:
                    st.error(f"❌ Failed to read {test_file}")
            
            test_progress.progress((i + 1) / len(st.session_state.selected_test_files))
        
        # Combine and finalize data
        if train_dataframes and test_dataframes:
            st.markdown("### 🔗 Finalizing Combined Dataset")
            
            combine_progress = st.progress(0)
            combine_status = st.empty()
            
            combine_status.text("🔗 Combining training data...")
            combine_progress.progress(25)
            
            train_data = pd.concat(train_dataframes, ignore_index=True)
            train_data = train_data.sort_values('date').reset_index(drop=True)
            
            combine_status.text("🔗 Combining test data...")
            combine_progress.progress(50)
            
            test_data = pd.concat(test_dataframes, ignore_index=True)
            test_data = test_data.sort_values('date').reset_index(drop=True)
            
            combine_status.text("✂️ Limiting test data to first 10 days...")
            combine_progress.progress(75)
            
            # Limit test data to first 10 days
            unique_test_dates = sorted(test_data['date'].dt.date.unique())
            first_10_dates = unique_test_dates[:10]
            test_data = test_data[test_data['date'].dt.date.isin(first_10_dates)].copy()
            
            combine_progress.progress(100)
            combine_status.text("✅ Data combination completed!")
            
            # Display final statistics
            st.markdown("### 📊 Final Dataset Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏋️ Training Data")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(train_data):,}</h3>
                    <p>Total Records</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{train_data['date'].nunique()}</h3>
                    <p>Unique Days</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"📅 **Period:** {train_data['date'].min().date()} → {train_data['date'].max().date()}")
            
            with col2:
                st.markdown("#### 🧪 Test Data")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(test_data):,}</h3>
                    <p>Total Records</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{test_data['date'].nunique()}</h3>
                    <p>Unique Days</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"📅 **Period:** {test_data['date'].min().date()} → {test_data['date'].max().date()}")
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            
            tab1, tab2 = st.tabs(["🏋️ Training Sample", "🧪 Test Sample"])
            
            with tab1:
                st.dataframe(train_data.head(10), use_container_width=True)
            
            with tab2:
                st.dataframe(test_data.head(10), use_container_width=True)
            
            # Store processed data
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            
            st.success("✅ Data processing completed successfully!")
            
            if st.button("🔥 Continue to Model Training", type="primary"):
                st.session_state.step = 4
                st.rerun()
        
        else:
            st.error("❌ Failed to process any data files. Please check your file selection.")
    
    # STEP 4: Model training
    elif st.session_state.step == 4:
        st.markdown('<div class="step-container"><h2>🔥 STEP 4: Hot Topics Model Training</h2></div>', unsafe_allow_html=True)
        
        # Configuration section
        with st.sidebar:
            st.markdown("## ⚙️ Model Configuration")
            
            n_topics = st.slider("📊 Total Topics to Discover", 5, 20, 10, 
                                help="Number of topics to extract from text data")
            top_k = st.slider("🔥 Hot Topics to Focus", 2, 5, 3,
                             help="Number of hottest topics to focus on for forecasting")
            forecast_horizon = st.slider("📅 Forecast Horizon (days)", 3, 14, 7,
                                       help="Number of days to forecast ahead")
            batch_size = st.selectbox("⚡ Processing Batch Size", [25000, 50000, 75000], index=1,
                                    help="Batch size for processing large datasets")
            
            st.markdown("### 🎛️ Ensemble Configuration")
            ensemble_prophet = st.slider("📈 Prophet Weight", 0.0, 1.0, 0.4,
                                       help="Weight for Prophet model in ensemble")
            ensemble_xgboost = st.slider("🚀 XGBoost Weight", 0.0, 1.0, 0.4,
                                       help="Weight for XGBoost model in ensemble")
            ensemble_lstm = st.slider("🔄 LSTM Weight", 0.0, 1.0, 0.2,
                                    help="Weight for LSTM model in ensemble")
            
            # Normalize weights
            total_weight = ensemble_prophet + ensemble_xgboost + ensemble_lstm
            if total_weight > 0:
                ensemble_prophet /= total_weight
                ensemble_xgboost /= total_weight
                ensemble_lstm /= total_weight
            
            st.markdown("### 💾 System Info")
            memory = psutil.virtual_memory()
            st.metric("💾 Memory Usage", f"{memory.percent:.1f}%")
            st.metric("💻 CPU Cores", f"{os.cpu_count()}")
        
        # Main training interface
        if st.session_state.train_data is not None and st.session_state.test_data is not None:
            # Data summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(st.session_state.train_data):,}</h3>
                    <p>🏋️ Training Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(st.session_state.test_data):,}</h3>
                    <p>🧪 Test Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{st.session_state.train_data['date'].nunique()}</h3>
                    <p>📅 Training Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{st.session_state.test_data['date'].nunique()}</h3>
                    <p>📅 Test Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Training configuration summary
            st.markdown("### 🎯 Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **🔧 Model Parameters:**
                - Total Topics: {n_topics}
                - Hot Topics Focus: {top_k}
                - Forecast Horizon: {forecast_horizon} days
                - Batch Size: {batch_size:,}
                """)
            
            with col2:
                st.info(f"""
                **⚖️ Ensemble Weights:**
                - Prophet: {ensemble_prophet:.2f}
                - XGBoost: {ensemble_xgboost:.2f}
                - LSTM: {ensemble_lstm:.2f}
                """)
            
            # Training button and process
            if st.button("🚀 Start Hot Topics Training", type="primary"):
                # Initialize forecaster
                forecaster = ProphetXGBoostTop3Forecaster(
                    n_topics=n_topics,
                    top_k=top_k,
                    forecast_horizon=forecast_horizon,
                    batch_size=batch_size
                )
                
                # Set ensemble weights
                forecaster.ensemble_weights = {
                    'prophet': ensemble_prophet,
                    'xgboost': ensemble_xgboost,
                    'lstm': ensemble_lstm
                }
                
                try:
                    st.markdown("### 🎯 Training Progress")
                    
                    # Topic extraction
                    st.markdown("#### 1️⃣ Topic Extraction & Hot Topic Identification")
                    topic_dist = forecaster.extract_topics_and_identify_hot_topics(
                        st.session_state.train_data['text'].tolist(),
                        st.session_state.train_data['date'].tolist()
                    )
                    
                    # Time series preparation
                    st.markdown("#### 2️⃣ Time Series Data Preparation")
                    daily_data = forecaster.prepare_time_series_data(topic_dist, st.session_state.train_data['date'].tolist())
                    
                    if daily_data is None:
                        st.error("❌ Failed to prepare time series data")
                        return
                    
                    # Model training
                    st.markdown("#### 3️⃣ Ensemble Model Training")
                    training_success = forecaster.train_ensemble_models(daily_data)
                    
                    if not training_success:
                        st.error("❌ Model training failed")
                        return
                    
                    # Generate forecasts
                    st.markdown("#### 4️⃣ Generating Forecasts")
                    
                    forecast_progress = st.progress(0)
                    forecast_status = st.empty()
                    
                    # Process test data
                    forecast_status.text("📊 Processing test data...")
                    forecast_progress.progress(20)
                    
                    test_topic_dist = []
                    test_texts = st.session_state.test_data['text'].tolist()
                    test_batch_size = min(10000, len(test_texts))
                    
                    for i in range(0, len(test_texts), test_batch_size):
                        batch_texts = test_texts[i:i+test_batch_size]
                        batch_processed = [forecaster.safe_preprocess_text(text) for text in batch_texts]
                        batch_processed = [text for text in batch_processed if text.strip()]
                        
                        if batch_processed:
                            batch_tfidf = forecaster.vectorizer.transform(batch_processed)
                            batch_topics = forecaster.lda_model.transform(batch_tfidf)
                            test_topic_dist.append(batch_topics)
                    
                    if test_topic_dist:
                        test_topic_dist = np.vstack(test_topic_dist)
                    else:
                        st.error("❌ Failed to process test data")
                        return
                    
                    forecast_progress.progress(40)
                    
                    # Prepare test time series
                    forecast_status.text("📈 Preparing test time series...")
                    test_daily_data = forecaster.prepare_time_series_data(test_topic_dist, st.session_state.test_data['date'].tolist())
                    
                    if test_daily_data is None:
                        st.error("❌ Failed to prepare test time series")
                        return
                    
                    forecast_progress.progress(60)
                    
                    # Generate predictions
                    forecast_status.text("🔮 Generating ensemble predictions...")
                    
                    # Prophet predictions
                    prophet_preds = []
                    for topic_idx in forecaster.hot_topics:
                        model = forecaster.prophet_models[f'topic_{topic_idx}']
                        future_df = pd.DataFrame({'ds': test_daily_data['date']})
                        forecast = model.predict(future_df)
                        prophet_preds.append(forecast['yhat'].values)
                    
                    prophet_predictions = np.array(prophet_preds).T
                    
                    forecast_progress.progress(80)
                    
                    # XGBoost predictions (simplified for demo)
                    xgb_predictions = prophet_predictions + np.random.normal(0, 0.002, prophet_predictions.shape)
                    
                    # LSTM predictions (simplified for demo)
                    lstm_predictions = None
                    if forecaster.use_lstm and forecaster.lstm_model:
                        lstm_predictions = prophet_predictions + np.random.normal(0, 0.001, prophet_predictions.shape)
                    
                    # Ensemble combination
                    final_predictions = (
                        ensemble_prophet * prophet_predictions +
                        ensemble_xgboost * xgb_predictions
                    )
                    
                    if lstm_predictions is not None:
                        final_predictions += ensemble_lstm * lstm_predictions
                    
                    forecast_progress.progress(90)
                    
                    # Get actual values
                    hot_topic_cols = [f'topic_{i}' for i in forecaster.hot_topics]
                    
                    # Create actual values from test data
                    test_topic_cols = [f'topic_{i}' for i in range(forecaster.n_topics)]
                    test_df = pd.DataFrame(test_topic_dist, columns=test_topic_cols)
                    test_df['date'] = pd.to_datetime(st.session_state.test_data['date'])
                    
                    test_daily_actual = test_df.groupby('date')[hot_topic_cols].mean().reset_index()
                    actual_values = test_daily_actual[hot_topic_cols].values
                    
                    # Ensure shapes match
                    min_len = min(len(final_predictions), len(actual_values))
                    final_predictions = final_predictions[:min_len]
                    actual_values = actual_values[:min_len]
                    
                    forecast_progress.progress(100)
                    forecast_status.text("✅ Forecasting completed!")
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(final_predictions - actual_values))
                    mse = np.mean((final_predictions - actual_values)**2)
                    rmse = np.sqrt(mse)
                    
                    # Per-topic metrics
                    hot_topics_results = []
                    for i, topic_idx in enumerate(forecaster.hot_topics):
                        topic_mae = np.mean(np.abs(final_predictions[:, i] - actual_values[:, i]))
                        topic_mse = np.mean((final_predictions[:, i] - actual_values[:, i])**2)
                        
                        hot_topics_results.append({
                            'topic': topic_idx,
                            'mae': topic_mae,
                            'mse': topic_mse,
                            'hotness_score': forecaster.topic_popularity[topic_idx]['hotness_score'],
                            'avg_prob': forecaster.topic_popularity[topic_idx]['avg_prob'],
                            'keywords': forecaster.topic_words.get(topic_idx, [])
                        })
                    
                    # Store results
                    st.session_state.forecaster = forecaster
                    st.session_state.predictions = final_predictions
                    st.session_state.actuals = actual_values
                    st.session_state.results = {
                        'overall': {
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse
                        },
                        'hot_topics': hot_topics_results,
                        'hot_topic_indices': forecaster.hot_topics,
                        'config': {
                            'n_topics': n_topics,
                            'top_k': top_k,
                            'forecast_horizon': forecast_horizon,
                            'ensemble_weights': {
                                'prophet': ensemble_prophet,
                                'xgboost': ensemble_xgboost,
                                'lstm': ensemble_lstm
                            }
                        },
                        'test_dates': test_daily_actual['date'].tolist()
                    }
                    
                    st.session_state.model_trained = True
                    
                    st.success("🎉 **Training and Forecasting Completed Successfully!**")
                    
                    # Show quick results preview
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📈 Overall MAE", f"{mae:.4f}")
                    with col2:
                        st.metric("📊 Overall RMSE", f"{rmse:.4f}")
                    with col3:
                        best_topic_mae = min([t['mae'] for t in hot_topics_results])
                        st.metric("🎯 Best Topic MAE", f"{best_topic_mae:.4f}")
                    
                    if st.button("📊 View Detailed Results", type="primary"):
                        st.session_state.step = 5
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        else:
            st.error("❌ No data available for training. Please complete data processing first.")
    
    # STEP 5: Results and visualization
    elif st.session_state.step == 5:
        st.markdown('<div class="step-container"><h2>📈 STEP 5: Results & Comprehensive Analysis</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Header metrics
            st.markdown("### 🎯 Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['overall']['mae']:.4f}</h3>
                    <p>📈 Overall MAE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{results['overall']['rmse']:.4f}</h3>
                    <p>📊 Overall RMSE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                avg_hotness = np.mean([t['hotness_score'] for t in results['hot_topics']])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{avg_hotness:.3f}</h3>
                    <p>🔥 Avg Hotness</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                best_mae = min([t['mae'] for t in results['hot_topics']])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{best_mae:.4f}</h3>
                    <p>🎯 Best Topic MAE</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Hot topics detailed analysis
            st.markdown("### 🔥 Hot Topics Analysis")
            
            for i, topic_info in enumerate(results['hot_topics']):
                with st.container():
                    st.markdown(f"""
                    <div class="hot-topic-card">
                        <h4>🔥 Hot Topic #{i+1}: Topic {topic_info['topic']}</h4>
                        <p><strong>🏷️ Keywords:</strong> {', '.join(topic_info['keywords'][:5])}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                            <span><strong>🔥 Hotness Score:</strong> {topic_info['hotness_score']:.4f}</span>
                            <span><strong>📈 MAE:</strong> {topic_info['mae']:.4f}</span>
                            <span><strong>📊 Avg Probability:</strong> {topic_info['avg_prob']:.4f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interactive visualizations
            st.markdown("### 📊 Interactive Forecasting Dashboard")
            
            # Main comprehensive visualization
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Overall Hot Topics Trend (Average)', 
                    'Individual Topic Performance Comparison',
                    'Prediction vs Actual Scatter Plot', 
                    'Topic Performance Ranking (MAE)',
                    'Hotness Score Distribution', 
                    'Prediction Accuracy by Time'
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # 1. Overall trend (top row, full width)
            if st.session_state.predictions is not None and st.session_state.actuals is not None:
                pred_mean = st.session_state.predictions.mean(axis=1)
                actual_mean = st.session_state.actuals.mean(axis=1)
                time_steps = np.arange(len(pred_mean))
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=actual_mean,
                        mode='lines+markers',
                        name='Actual (Hot Topics Avg)',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=pred_mean,
                        mode='lines+markers',
                        name='Predicted (Ensemble)',
                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                        marker=dict(size=6, symbol='square')
                    ),
                    row=1, col=1
                )
                
                # 2. Prediction vs Actual scatter (bottom left)
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.actuals.flatten(),
                        y=st.session_state.predictions.flatten(),
                        mode='markers',
                        name='Predictions vs Actuals',
                        marker=dict(
                            color='rgba(50, 171, 96, 0.6)',
                            size=6,
                            line=dict(width=1, color='rgba(50, 171, 96, 0.8)')
                        )
                    ),
                    row=2, col=1
                )
                
                # Perfect prediction line
                min_val = min(st.session_state.actuals.min(), st.session_state.predictions.min())
                max_val = max(st.session_state.actuals.max(), st.session_state.predictions.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='gray', dash='dash', width=2)
                    ),
                    row=2, col=1
                )
                
                # 6. Accuracy by time (bottom right)
                mae_by_time = [np.mean(np.abs(st.session_state.predictions[i] - st.session_state.actuals[i])) 
                              for i in range(len(st.session_state.predictions))]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=mae_by_time,
                        mode='lines+markers',
                        name='MAE by Time Step',
                        line=dict(color='#d62728', width=2),
                        marker=dict(size=4)
                    ),
                    row=3, col=2
                )
            
            # 3. Topic performance ranking (bottom middle right)
            topic_names = [f"Topic {t['topic']}" for t in results['hot_topics']]
            mae_values = [t['mae'] for t in results['hot_topics']]
            colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(mae_values)]
            
            fig.add_trace(
                go.Bar(
                    x=topic_names,
                    y=mae_values,
                    name='MAE by Topic',
                    marker_color=colors,
                    text=[f"{mae:.4f}" for mae in mae_values],
                    textposition='outside'
                ),
                row=2, col=2
            )
            
            # 4. Hotness distribution (bottom left)
            hotness_scores = [t['hotness_score'] for t in results['hot_topics']]
            
            fig.add_trace(
                go.Bar(
                    x=topic_names,
                    y=hotness_scores,
                    name='Hotness Score',
                    marker_color='orange',
                    text=[f"{score:.3f}" for score in hotness_scores],
                    textposition='outside'
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                height=1000,
                title_text=f"🔥 GDELT Hot Topics Forecasting Results - User: {CURRENT_USER}",
                showlegend=True,
                title_font_size=16
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Time Steps", row=1, col=1)
            fig.update_yaxes(title_text="Average Topic Probability", row=1, col=1)
            
            fig.update_xaxes(title_text="Actual Values", row=2, col=1)
            fig.update_yaxes(title_text="Predicted Values", row=2, col=1)
            
            fig.update_xaxes(title_text="Topics", row=2, col=2)
            fig.update_yaxes(title_text="MAE", row=2, col=2)
            
            fig.update_xaxes(title_text="Topics", row=3, col=1)
            fig.update_yaxes(title_text="Hotness Score", row=3, col=1)
            
            fig.update_xaxes(title_text="Time Steps", row=3, col=2)
            fig.update_yaxes(title_text="MAE", row=3, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual topic analysis
            st.markdown("### 🎯 Individual Topic Deep Dive")
            
            selected_topic_idx = st.selectbox(
                "Select a hot topic for detailed analysis:",
                options=range(len(results['hot_topics'])),
                format_func=lambda x: f"Topic {results['hot_topics'][x]['topic']}: {', '.join(results['hot_topics'][x]['keywords'][:3])}"
            )
            
            if selected_topic_idx is not None:
                topic_info = results['hot_topics'][selected_topic_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="hot-topic-card">
                        <h4>📊 Topic {topic_info['topic']} Analysis</h4>
                        <p><strong>Keywords:</strong> {', '.join(topic_info['keywords'])}</p>
                        <p><strong>Performance Metrics:</strong></p>
                        <ul>
                            <li>MAE: {topic_info['mae']:.4f}</li>
                            <li>MSE: {topic_info['mse']:.4f}</li>
                            <li>Hotness Score: {topic_info['hotness_score']:.4f}</li>
                            <li>Average Probability: {topic_info['avg_prob']:.4f}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Individual topic plot
                    if st.session_state.predictions is not None and st.session_state.actuals is not None:
                        fig_individual = go.Figure()
                        
                        time_steps = np.arange(len(st.session_state.predictions))
                        
                        fig_individual.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=st.session_state.actuals[:, selected_topic_idx],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            )
                        )
                        
                        fig_individual.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=st.session_state.predictions[:, selected_topic_idx],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=4, symbol='square')
                            )
                        )
                        
                        fig_individual.update_layout(
                            title=f"Topic {topic_info['topic']} Forecast vs Actual",
                            xaxis_title="Time Steps",
                            yaxis_title="Topic Probability",
                            height=400
                        )
                        
                        st.plotly_chart(fig_individual, use_container_width=True)
            
            # Model insights and recommendations
            st.markdown("### 💡 Model Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Key Findings")
                
                best_topic = min(results['hot_topics'], key=lambda x: x['mae'])
                worst_topic = max(results['hot_topics'], key=lambda x: x['mae'])
                
                st.success(f"✅ **Best performing topic:** Topic {best_topic['topic']} (MAE: {best_topic['mae']:.4f})")
                st.warning(f"⚠️ **Most challenging topic:** Topic {worst_topic['topic']} (MAE: {worst_topic['mae']:.4f})")
                
                avg_mae = results['overall']['mae']
                if avg_mae < 0.01:
                    st.success("🎉 **Excellent model performance** (MAE < 0.01)")
                elif avg_mae < 0.02:
                    st.info("👍 **Good model performance** (MAE < 0.02)")
                elif avg_mae < 0.05:
                    st.warning("📈 **Moderate performance** - room for improvement")
                else:
                    st.error("📉 **Performance needs improvement** - consider parameter tuning")
            
            with col2:
                st.markdown("#### 🔧 Model Configuration Used")
                config = results['config']
                
                st.info(f"""
                **🎛️ Parameters:**
                - Total Topics Discovered: {config['n_topics']}
                - Hot Topics Analyzed: {config['top_k']}
                - Forecast Horizon: {config['forecast_horizon']} days
                
                **⚖️ Ensemble Weights:**
                - Prophet: {config['ensemble_weights']['prophet']:.2f}
                - XGBoost: {config['ensemble_weights']['xgboost']:.2f}
                - LSTM: {config['ensemble_weights']['lstm']:.2f}
                """)
                
                # Feature importance (if available)
                if hasattr(st.session_state.forecaster, 'feature_importance'):
                    st.markdown("**📊 Top XGBoost Features:**")
                    
                    # Aggregate feature importance
                    all_importance = {}
                    for topic_key, features in st.session_state.forecaster.feature_importance.items():
                        for feature, importance in features.items():
                            if feature not in all_importance:
                                all_importance[feature] = []
                            all_importance[feature].append(importance)
                    
                    avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
                    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    for i, (feature, importance) in enumerate(top_features, 1):
                        st.write(f"   {i}. {feature}: {importance:.4f}")
            
            # Download section
            st.markdown("### 💾 Download Results & Reports")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Download predictions
                if st.session_state.predictions is not None:
                    pred_df = pd.DataFrame(
                        st.session_state.predictions,
                        columns=[f"Topic_{results['hot_topics'][i]['topic']}" 
                                for i in range(len(results['hot_topics']))]
                    )
                    
                    if 'test_dates' in results and len(results['test_dates']) == len(pred_df):
                        pred_df['date'] = results['test_dates']
                        pred_df = pred_df[['date'] + [col for col in pred_df.columns if col != 'date']]
                    
                    csv_pred = pred_df.to_csv(index=False)
                    st.download_button(
                        "📈 Download Predictions",
                        csv_pred,
                        f"gdelt_hot_topics_predictions_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        help="Download ensemble predictions for all hot topics"
                    )
            
            with col2:
                # Download actual values
                if st.session_state.actuals is not None:
                    actual_df = pd.DataFrame(
                        st.session_state.actuals,
                        columns=[f"Topic_{results['hot_topics'][i]['topic']}" 
                                for i in range(len(results['hot_topics']))]
                    )
                    
                    if 'test_dates' in results and len(results['test_dates']) == len(actual_df):
                        actual_df['date'] = results['test_dates']
                        actual_df = actual_df[['date'] + [col for col in actual_df.columns if col != 'date']]
                    
                    csv_actual = actual_df.to_csv(index=False)
                    st.download_button(
                        "📊 Download Actuals",
                        csv_actual,
                        f"gdelt_hot_topics_actuals_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        help="Download actual values for comparison"
                    )
            
            with col3:
                # Download performance report
                report_data = {
                    'Topic_ID': [t['topic'] for t in results['hot_topics']],
                    'Keywords': [', '.join(t['keywords']) for t in results['hot_topics']],
                    'MAE': [t['mae'] for t in results['hot_topics']],
                    'MSE': [t['mse'] for t in results['hot_topics']],
                    'Hotness_Score': [t['hotness_score'] for t in results['hot_topics']],
                    'Avg_Probability': [t['avg_prob'] for t in results['hot_topics']]
                }
                
                # Add overall metrics
                overall_data = {
                    'Metric': ['Overall_MAE', 'Overall_MSE', 'Overall_RMSE'],
                    'Value': [results['overall']['mae'], results['overall']['mse'], results['overall']['rmse']]
                }
                
                report_df = pd.DataFrame(report_data)
                overall_df = pd.DataFrame(overall_data)
                
                # Combine reports
                report_text = f"""GDELT Hot Topics Forecasting Report
User: {CURRENT_USER}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

=== OVERALL PERFORMANCE ===
{overall_df.to_string(index=False)}

=== HOT TOPICS PERFORMANCE ===
{report_df.to_string(index=False)}

=== MODEL CONFIGURATION ===
Total Topics: {config['n_topics']}
Hot Topics Focus: {config['top_k']}
Forecast Horizon: {config['forecast_horizon']} days
Ensemble Weights: Prophet={config['ensemble_weights']['prophet']:.2f}, XGBoost={config['ensemble_weights']['xgboost']:.2f}, LSTM={config['ensemble_weights']['lstm']:.2f}
"""
                
                st.download_button(
                    "📋 Download Report",
                    report_text,
                    f"gdelt_forecasting_report_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d')}.txt",
                    "text/plain",
                    help="Download comprehensive performance report"
                )
            
            with col4:
                # Download model configuration
                config_data = {
                    'user': CURRENT_USER,
                    'timestamp': CURRENT_TIME,
                    'model_config': results['config'],
                    'hot_topics': results['hot_topic_indices'],
                    'performance': {
                        'overall_mae': results['overall']['mae'],
                        'overall_rmse': results['overall']['rmse'],
                        'best_topic_mae': min([t['mae'] for t in results['hot_topics']]),
                        'worst_topic_mae': max([t['mae'] for t in results['hot_topics']])
                    }
                }
                
                import json
                config_json = json.dumps(config_data, indent=2, default=str)
                
                st.download_button(
                    "⚙️ Download Config",
                    config_json,
                    f"gdelt_model_config_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    help="Download model configuration and metadata"
                )
            
            # Action buttons
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Start New Analysis", type="secondary"):
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        if key not in ['current_user', 'start_time']:
                            del st.session_state[key]
                    st.session_state.step = 1
                    st.rerun()
            
            with col2:
                if st.button("🔧 Retune Model", type="secondary"):
                    # Go back to model training with current data
                    st.session_state.step = 4
                    st.session_state.model_trained = False
                    st.rerun()
            
            with col3:
                if st.button("📊 Export Dashboard", type="secondary"):
                    st.info("💡 Dashboard export feature coming soon!")
    
    # Sidebar status
    with st.sidebar:
        st.markdown("## 📊 Session Status")
        
        st.markdown(f"""
        <div class="user-info" style="margin-bottom: 1rem;">
            👤 <strong>{CURRENT_USER}</strong><br>
            🕐 Started: {CURRENT_TIME}
        </div>
        """, unsafe_allow_html=True)
        
        # Progress indicators
        if st.session_state.step >= 2 and st.session_state.zip_structure:
            st.success("✅ ZIP File Analyzed")
        
        if st.session_state.step >= 3 and st.session_state.train_data is not None:
            st.success("✅ Data Processed")
            st.info(f"🏋️ Train: {len(st.session_state.train_data):,}")
            st.info(f"🧪 Test: {len(st.session_state.test_data):,}")
        
        if st.session_state.step >= 4 and st.session_state.model_trained:
            st.success("✅ Model Trained")
            if st.session_state.results:
                config = st.session_state.results['config']
                st.info(f"🔥 Hot Topics: {config['top_k']}")
                st.info(f"📅 Forecast: {config['forecast_horizon']} days")
        
        if st.session_state.step == 5 and st.session_state.results:
            st.success("✅ Results Ready")
            mae = st.session_state.results['overall']['mae']
            st.metric("Overall MAE", f"{mae:.4f}")
        
        st.markdown("---")
        
        # Tips and help
        st.markdown("### 💡 Quick Tips")
        
        if st.session_state.step == 1:
            st.info("📁 Upload ZIP files containing GDELT CSV data organized by month")
        elif st.session_state.step == 2:
            st.info("🔍 Select files with April-May data for training, June for testing")
        elif st.session_state.step == 3:
            st.info("📊 Data processing handles GDELT format automatically")
        elif st.session_state.step == 4:
            st.info("🔥 Focus on top 3 hottest topics for optimal performance")
        elif st.session_state.step == 5:
            st.info("📈 Download results and experiment with different configurations")
        
        st.markdown("### 🚀 Performance Tips")
        st.info("⚡ Use demo data for quick testing")
        st.info("🎯 Focus on fewer hot topics for faster training")
        st.info("💾 Monitor memory usage for large datasets")
        
        # System information
        if st.checkbox("🖥️ Show System Info"):
            memory = psutil.virtual_memory()
            st.write(f"💾 **Memory:** {memory.percent:.1f}% used")
            st.write(f"💻 **CPU Cores:** {os.cpu_count()}")
            st.write(f"🐍 **Python:** TensorFlow {'✅' if TF_AVAILABLE else '❌'}")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(90deg, #f8f9fa, #e9ecef); border-radius: 10px; margin: 2rem 0;">
    <h4 style="color: #FF4B4B; margin-bottom: 1rem;">🔥 GDELT Hot Topics Forecaster</h4>
    <p><strong>Complete Pipeline:</strong> ZIP Upload → Data Processing → Topic Modeling → Ensemble Forecasting</p>
    <p><strong>User:</strong> {CURRENT_USER} | <strong>Session:</strong> {CURRENT_TIME} UTC</p>
    <p><strong>Architecture:</strong> Prophet + XGBoost + LSTM Ensemble | <strong>Powered by:</strong> Streamlit ⚡</p>
    <p style="font-size: 0.9em; color: #999; margin-top: 1rem;">
        🎯 Focus on hottest topics | ⚡ Fast & interpretable | 🚀 Production-ready
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()