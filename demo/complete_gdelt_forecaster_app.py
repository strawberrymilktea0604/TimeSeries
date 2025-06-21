Dựa trên thông tin bạn cung cấp, tôi sẽ cập nhật toàn bộ code với thời gian và user hiện tại, cùng với tất cả các tối ưu hóa cho việc xử lý upload ZIP file:

```python
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

# Current user and time - UPDATED
CURRENT_USER = "strawberrymilktea0604"
CURRENT_TIME = "2025-06-21 12:34:35"

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
    .warning-card {
        background: linear-gradient(135deg, #FFF3CD, #FCE4EC);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #D4EDDA, #C3E6CB);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28A745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptimizedGDELTDataProcessor:
    """Enhanced GDELT Data Processor with comprehensive error handling and memory management"""
    
    def __init__(self):
        self.temp_dir = None
        self.max_file_size_mb = 150  # Reduced for stability
        self.max_files_to_process = 15  # Reduced for performance
        self.max_records_per_file = 25000  # Memory-safe limit
        
    def safe_explore_zip_structure(self, zip_file):
        """Safe ZIP exploration with comprehensive error handling"""
        st.write("🔍 **Safely analyzing ZIP file structure...**")
        
        try:
            # Check file size first
            file_size_mb = len(zip_file.getvalue()) / (1024 * 1024)
            
            if file_size_mb > self.max_file_size_mb:
                st.error(f"❌ File too large: {file_size_mb:.1f}MB. Maximum allowed: {self.max_file_size_mb}MB")
                st.info("💡 **Tip:** Try splitting your ZIP file into smaller parts or use demo data to test the system.")
                return None
            
            st.markdown(f"""
            <div class="success-card">
                <h4>✅ File Size Check Passed</h4>
                <p>File size: <strong>{file_size_mb:.1f}MB</strong> (Limit: {self.max_file_size_mb}MB)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Step 1: Read ZIP file
            status_text.text("📂 Reading ZIP file...")
            progress_bar.progress(10)
            
            zip_buffer = io.BytesIO(zip_file.getvalue())
            
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                status_text.text("📋 Listing files...")
                progress_bar.progress(25)
                
                file_list = zf.namelist()
                
                # Filter and validate files
                status_text.text("🔍 Filtering CSV files...")
                progress_bar.progress(40)
                
                csv_files = []
                skipped_files = []
                
                for file_path in file_list:
                    if file_path.endswith('.csv'):
                        try:
                            file_info = zf.getinfo(file_path)
                            file_size_mb = file_info.file_size / (1024 * 1024)
                            
                            # Skip very large individual files
                            if file_size_mb > 30:  # 30MB per file limit
                                skipped_files.append(f"{file_path} (too large: {file_size_mb:.1f}MB)")
                                continue
                            
                            # Skip empty files
                            if file_info.file_size == 0:
                                skipped_files.append(f"{file_path} (empty file)")
                                continue
                                
                            csv_files.append(file_path)
                            
                        except Exception as e:
                            skipped_files.append(f"{file_path} (error: {str(e)[:50]})")
                            continue
                
                # Limit number of files
                if len(csv_files) > self.max_files_to_process:
                    st.warning(f"⚠️ Found {len(csv_files)} CSV files. Limiting to first {self.max_files_to_process} for performance.")
                    csv_files = csv_files[:self.max_files_to_process]
                
                status_text.text("🏷️ Categorizing files...")
                progress_bar.progress(60)
                
                # Analyze file names
                file_analysis = self.safe_analyze_file_names(csv_files)
                
                status_text.text("📊 Creating directory structure...")
                progress_bar.progress(80)
                
                directories = list(set([os.path.dirname(f) for f in file_list if os.path.dirname(f)]))
                directories = sorted(directories)
                
                progress_bar.progress(100)
                status_text.text("✅ ZIP analysis completed!")
                
                # Display results with enhanced UI
                st.markdown("### 📊 Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(file_list)}</h3>
                        <p>📁 Total Files</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(csv_files)}</h3>
                        <p>📊 CSV Files</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(file_analysis['train_candidates'])}</h3>
                        <p>🏋️ Training</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(file_analysis['test_candidates'])}</h3>
                        <p>🧪 Testing</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show file categorization
                with st.expander("📂 File Categorization Details", expanded=True):
                    tab1, tab2, tab3 = st.tabs(["🏋️ Training Files", "🧪 Test Files", "❓ Unknown Files"])
                    
                    with tab1:
                        if file_analysis['train_candidates']:
                            for file in file_analysis['train_candidates']:
                                st.write(f"✅ `{os.path.basename(file)}`")
                        else:
                            st.warning("⚠️ No training candidates found")
                    
                    with tab2:
                        if file_analysis['test_candidates']:
                            for file in file_analysis['test_candidates']:
                                st.write(f"✅ `{os.path.basename(file)}`")
                        else:
                            st.warning("⚠️ No test candidates found")
                    
                    with tab3:
                        if file_analysis['unknown_files']:
                            for file in file_analysis['unknown_files'][:10]:
                                st.write(f"❓ `{os.path.basename(file)}`")
                            if len(file_analysis['unknown_files']) > 10:
                                st.write(f"... and {len(file_analysis['unknown_files']) - 10} more files")
                        else:
                            st.info("✅ All files categorized")
                
                if skipped_files:
                    with st.expander(f"⚠️ Skipped Files ({len(skipped_files)})", expanded=False):
                        for skipped in skipped_files[:10]:
                            st.write(f"• {skipped}")
                        if len(skipped_files) > 10:
                            st.write(f"... and {len(skipped_files) - 10} more files")
                
                return {
                    'csv_files': csv_files,
                    'directories': directories,
                    'file_analysis': file_analysis,
                    'total_files': len(file_list),
                    'skipped_files': skipped_files,
                    'file_size_mb': file_size_mb,
                    'processing_stats': {
                        'max_file_size_mb': self.max_file_size_mb,
                        'max_files_to_process': self.max_files_to_process,
                        'timestamp': CURRENT_TIME,
                        'user': CURRENT_USER
                    }
                }
                
        except zipfile.BadZipFile:
            st.error("❌ Invalid ZIP file format. Please check your file.")
            st.info("💡 **Troubleshooting:** Ensure the file is a valid ZIP archive and not corrupted.")
            return None
        except MemoryError:
            st.error("❌ Not enough memory to process this ZIP file. Try a smaller file.")
            st.info("💡 **Solution:** Use demo data or split your ZIP file into smaller parts.")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error reading ZIP file: {str(e)}")
            
            # Enhanced debug information
            with st.expander("🔍 Debug Information", expanded=False):
                import traceback
                st.code(traceback.format_exc())
                
                st.write("**System Information:**")
                st.write(f"- Memory usage: {psutil.virtual_memory().percent:.1f}%")
                st.write(f"- Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
                st.write(f"- CPU cores: {psutil.cpu_count()}")
                st.write(f"- User: {CURRENT_USER}")
                st.write(f"- Timestamp: {CURRENT_TIME}")
            
            return None
    
    def safe_analyze_file_names(self, csv_files):
        """Enhanced file name analysis with better pattern matching"""
        analysis = {
            'train_candidates': [],
            'test_candidates': [],
            'unknown_files': []
        }
        
        # Enhanced pattern matching
        train_patterns = [
            'april', 'apr', 'may', '04', '05', 
            'tháng4', 'tháng5', '2024-04', '2024-05',
            'train', 'training', '202404', '202405'
        ]
        
        test_patterns = [
            'june', 'jun', '06', 'tháng6', '2024-06',
            'test', 'testing', 'validation', 'val', '202406'
        ]
        
        for file_path in csv_files:
            filename_lower = file_path.lower()
            basename = os.path.basename(filename_lower)
            
            # Check for training patterns
            if any(pattern in filename_lower for pattern in train_patterns):
                analysis['train_candidates'].append(file_path)
            # Check for test patterns
            elif any(pattern in filename_lower for pattern in test_patterns):
                analysis['test_candidates'].append(file_path)
            else:
                analysis['unknown_files'].append(file_path)
        
        return analysis
    
    def safe_extract_and_read_csv(self, zip_file, csv_filename, max_rows=None):
        """Safe CSV extraction with memory limits and encoding detection"""
        
        st.write(f"📄 **Reading:** `{os.path.basename(csv_filename)}`")
        
        try:
            # Progress tracking
            read_progress = st.progress(0)
            read_status = st.empty()
            
            read_status.text("📂 Extracting from ZIP...")
            read_progress.progress(20)
            
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Check file size in ZIP
                file_info = zf.getinfo(csv_filename)
                file_size_mb = file_info.file_size / (1024 * 1024)
                
                if file_size_mb > 30:  # 30MB limit per file
                    st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB) - will sample data")
                    max_rows = min(max_rows or self.max_records_per_file, self.max_records_per_file)
                
                read_status.text("🔍 Detecting encoding and format...")
                read_progress.progress(40)
                
                # Read raw content
                csv_content = zf.read(csv_filename)
                
                # Try different encodings and separators
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
                separators = ['\t', ',', ';', '|']
                
                best_df = None
                best_info = None
                
                for encoding in encodings:
                    for separator in separators:
                        try:
                            read_status.text(f"🧪 Testing {encoding} with '{separator}'...")
                            
                            # Decode content
                            decoded_content = csv_content.decode(encoding)
                            
                            # Try to read CSV
                            df = pd.read_csv(
                                io.StringIO(decoded_content),
                                sep=separator,
                                dtype=str,
                                low_memory=False,
                                on_bad_lines='skip',
                                nrows=max_rows,
                                encoding=None  # Already decoded
                            )
                            
                            # Validate DataFrame
                            if len(df.columns) > 1 and len(df) > 0:
                                # Score this attempt
                                score = len(df.columns) * len(df)
                                
                                if best_df is None or score > best_info['score']:
                                    best_df = df
                                    best_info = {
                                        'encoding': encoding,
                                        'separator': separator,
                                        'score': score,
                                        'shape': df.shape
                                    }
                                
                                # If we have a good result, we can break
                                if len(df.columns) >= 10 and len(df) >= 100:
                                    break
                                    
                        except Exception:
                            continue
                    
                    if best_df is not None and best_info['score'] > 1000:
                        break
                
                read_progress.progress(80)
                
                if best_df is not None:
                    read_status.text("✅ Successfully read CSV!")
                    read_progress.progress(100)
                    
                    # Display success info with enhanced UI
                    st.markdown(f"""
                    <div class="success-card">
                        <h4>✅ CSV Read Successfully</h4>
                        <p><strong>Encoding:</strong> {best_info['encoding']} | 
                           <strong>Separator:</strong> '{best_info['separator']}' | 
                           <strong>Shape:</strong> {best_info['shape']}</p>
                        <p><strong>Columns:</strong> {len(best_df.columns)} | 
                           <strong>Records:</strong> {len(best_df):,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    return best_df, best_info['separator'], best_info['encoding']
                else:
                    read_status.text("❌ Failed to read CSV")
                    st.error("❌ Could not read CSV with any encoding/separator combination")
                    
                    # Enhanced troubleshooting
                    st.markdown("""
                    <div class="warning-card">
                        <h4>🔧 Troubleshooting Tips</h4>
                        <ul>
                            <li>Check if the file is actually a CSV format</li>
                            <li>Ensure the file is not corrupted</li>
                            <li>Try opening the file in a text editor to check format</li>
                            <li>Consider using a different file or demo data</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    return None, None, None
                
        except Exception as e:
            st.error(f"❌ Error reading file {csv_filename}: {str(e)}")
            return None, None, None
    
    def safe_process_gdelt_dataframe(self, df, file_type="unknown", max_records=None):
        """Safe GDELT DataFrame processing with memory management"""
        
        st.write(f"🔧 **Processing {file_type} data safely...**")
        
        try:
            # Initial info
            original_shape = df.shape
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Original Shape", f"{original_shape[0]:,} × {original_shape[1]}")
            with col2:
                st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")
            
            # Progress tracking
            process_progress = st.progress(0)
            process_status = st.empty()
            
            # Step 1: Column validation
            process_status.text("🔍 Validating columns...")
            process_progress.progress(10)
            
            # Show available columns in a nice format
            with st.expander("📋 Available Columns", expanded=False):
                cols = list(df.columns)
                for i in range(0, len(cols), 4):
                    col_batch = cols[i:i+4]
                    col_str = " | ".join([f"`{col}`" for col in col_batch])
                    st.write(col_str)
            
            # Find required columns with enhanced matching
            date_col = None
            theme_col = None
            
            # Look for date column
            date_candidates = ['DATE', 'Date', 'date', 'SQLDATE', 'EventDate', 'Day', 'day']
            for col in date_candidates:
                if col in df.columns:
                    date_col = col
                    break
            
            # Look for theme/content column
            theme_candidates = ['THEMES', 'Themes', 'themes', 'V2Themes', 'Actor1Name', 'Actor2Name', 'EventRootCode']
            for col in theme_candidates:
                if col in df.columns:
                    theme_col = col
                    break
            
            if not date_col:
                st.error("❌ No date column found")
                st.info(f"💡 Looking for: {', '.join(date_candidates)}")
                return None
                
            if not theme_col:
                st.error("❌ No theme/content column found")
                st.info(f"💡 Looking for: {', '.join(theme_candidates)}")
                return None
            
            st.success(f"✅ Using DATE column: `{date_col}`, THEME column: `{theme_col}`")
            
            # Step 2: Sample data if too large
            process_status.text("📊 Checking data size...")
            process_progress.progress(25)
            
            if max_records and len(df) > max_records:
                st.warning(f"⚠️ Large dataset ({len(df):,} rows). Sampling {max_records:,} rows for performance.")
                df = df.sample(n=max_records, random_state=42).reset_index(drop=True)
            
            # Step 3: Process dates
            process_status.text("📅 Converting dates...")
            process_progress.progress(40)
            
            # Try different date formats
            date_formats = ['%Y%m%d', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d%H%M%S']
            df['date'] = None
            
            for date_format in date_formats:
                try:
                    if date_format == '%Y%m%d%H%M%S':
                        # Handle datetime format by taking first 8 characters
                        df['date'] = pd.to_datetime(df[date_col].astype(str).str[:8], format='%Y%m%d', errors='coerce')
                    else:
                        df['date'] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
                    
                    valid_dates = df['date'].notna().sum()
                    if valid_dates > len(df) * 0.5:  # At least 50% valid dates
                        st.info(f"✅ Date format detected: `{date_format}`")
                        break
                except:
                    continue
            
            # If no format worked, try automatic parsing
            if df['date'].notna().sum() < len(df) * 0.1:
                df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Step 4: Process themes
            process_status.text("🎯 Processing themes...")
            process_progress.progress(60)
            
            df['themes_text'] = df[theme_col].fillna('').astype(str)
            
            # Handle different theme formats
            def safe_process_themes(theme_text):
                try:
                    if pd.isna(theme_text) or theme_text == '':
                        return []
                    
                    theme_text = str(theme_text)
                    
                    # Handle semicolon-separated themes
                    if ';' in theme_text:
                        themes = [t.strip() for t in theme_text.split(';') if t.strip()]
                    # Handle comma-separated themes
                    elif ',' in theme_text:
                        themes = [t.strip() for t in theme_text.split(',') if t.strip()]
                    # Single theme
                    else:
                        themes = [theme_text.strip()] if theme_text.strip() else []
                    
                    return themes[:10]  # Limit to 10 themes per record
                    
                except:
                    return []
            
            df['themes_list'] = df['themes_text'].apply(safe_process_themes)
            
            # Step 5: Create text content
            process_status.text("📝 Creating text content...")
            process_progress.progress(80)
            
            def safe_create_text(themes_list):
                try:
                    if not themes_list:
                        return ""
                    
                    # Clean and process themes
                    processed_themes = []
                    for theme in themes_list:
                        # Replace underscores with spaces, convert to lowercase
                        clean_theme = theme.replace('_', ' ').lower().strip()
                        if len(clean_theme) > 2:  # Minimum length
                            processed_themes.append(clean_theme)
                    
                    return ' '.join(processed_themes)
                    
                except:
                    return ""
            
            df['text'] = df['themes_list'].apply(safe_create_text)
            
            # Step 6: Clean and validate
            process_status.text("🧹 Final cleaning...")
            process_progress.progress(90)
            
            # Remove invalid records
            initial_count = len(df)
            df = df.dropna(subset=['date'])
            df = df[df['text'].str.strip() != '']
            df = df[df['text'].str.len() > 5]  # Minimum text length
            
            final_count = len(df)
            
            if final_count < initial_count * 0.1:  # Less than 10% data remaining
                st.warning(f"⚠️ Only {final_count:,}/{initial_count:,} records remain after cleaning. Data quality may be poor.")
            
            # Create final dataset
            result_df = df[['date', 'text']].copy()
            result_df = result_df.sort_values('date').reset_index(drop=True)
            
            process_progress.progress(100)
            process_status.text("✅ Processing completed!")
            
            # Show results with enhanced metrics
            st.markdown("### 📊 Processing Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(result_df):,}</h3>
                    <p>📊 Final Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                date_range = (result_df['date'].max() - result_df['date'].min()).days
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{date_range}</h3>
                    <p>📅 Days Span</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                daily_avg = result_df.groupby(result_df['date'].dt.date).size().mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{daily_avg:.1f}</h3>
                    <p>📈 Avg/Day</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                retention_rate = (final_count / initial_count) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{retention_rate:.1f}%</h3>
                    <p>🎯 Retention</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show date range and sample data
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"📅 **Date Range:** {result_df['date'].min().date()} → {result_df['date'].max().date()}")
            
            with col2:
                avg_text_length = result_df['text'].str.len().mean()
                st.info(f"📝 **Avg Text Length:** {avg_text_length:.1f} characters")
            
            # Show sample data
            with st.expander("👀 Sample Processed Data", expanded=False):
                st.dataframe(result_df.head(10), use_container_width=True)
            
            return result_df
            
        except Exception as e:
            st.error(f"❌ Data processing failed: {str(e)}")
            
            # Enhanced debug information
            with st.expander("🔍 Debug Information", expanded=False):
                import traceback
                st.code(traceback.format_exc())
                
                st.write("**Processing Context:**")
                st.write(f"- File type: {file_type}")
                st.write(f"- Original shape: {original_shape}")
                st.write(f"- User: {CURRENT_USER}")
                st.write(f"- Timestamp: {CURRENT_TIME}")
            
            return None
    
    def create_demo_data(self):
        """Create realistic demo data with current timestamp"""
        st.write("🎭 **Creating demo data...**")
        
        np.random.seed(42)
        
        # Enhanced GDELT themes with more realistic content
        gdelt_themes = [
            'TRIAL TAX_FNCACT TAX_FNCACT_LAWYER LEGAL_PROCEEDING',
            'WB_1979_NATURAL_RESOURCE_MANAGEMENT WB_435_AGRICULTURE_AND_FOOD_SECURITY ENVIRONMENT',
            'PORTSMEN_HOLIDAY CRISISLEX_CRISISLEXREC SOC_POINTSOFINTEREST TOURISM',
            'TAX_FNCACT_POLICE SOC_POINTSOFINTEREST_PRISON WB_2405_DETENTION_REFORM SECURITY',
            'ARREST TAX_FNCACT TAX_FNCACT_OFFICIALS TRIAL JUSTICE',
            'TERROR ARMEDCONFLICT TAX_ETHNICITY_VENEZUELANS CONFLICT',
            'WB_826_TOURISM WB_1921_COMPETITIVE_AND_REAL_SECTORS ECONOMY',
            'EPU_ECONOMY EPU_ECONOMY_HISTORIC TAX_ETHNICITY_SPANISH POLITICS',
            'WB_698 MEDIA_MSM AFFECT BAN INFORMATION',
            'SECURITY_SERVICES CRIME WB_ILLEGAL_DRUGS LAW_ENFORCEMENT',
            'CLIMATE_CHANGE ENVIRONMENT_PROTECTION SUSTAINABILITY',
            'TECHNOLOGY_INNOVATION DIGITAL_TRANSFORMATION AI_DEVELOPMENT',
            'HEALTHCARE_REFORM MEDICAL_BREAKTHROUGH PANDEMIC_RESPONSE',
            'EDUCATION_POLICY STUDENT_PROTESTS ACADEMIC_FREEDOM',
            'TRADE_AGREEMENT INTERNATIONAL_COMMERCE SUPPLY_CHAIN'
        ]
        
        # Generate training data (April-May 2024)
        dates_train = pd.date_range('2024-04-01', '2024-05-31', freq='D')
        train_data = []
        
        progress_bar = st.progress(0)
        
        for i, date in enumerate(dates_train):
            n_articles = np.random.randint(100, 250)  # More realistic article count
            for _ in range(n_articles):
                # Create more realistic theme combinations
                n_themes = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                selected_themes = np.random.choice(gdelt_themes, n_themes, replace=False)
                
                # Create text with theme processing
                text_parts = []
                for theme in selected_themes:
                    processed_theme = theme.replace('_', ' ').lower()
                    text_parts.append(processed_theme)
                
                text = ' '.join(text_parts)
                train_data.append({'date': date, 'text': text})
            
            progress_bar.progress((i + 1) / len(dates_train) * 0.7)
        
        # Generate test data (June 2024)
        dates_test = pd.date_range('2024-06-01', '2024-06-10', freq='D')
        test_data = []
        
        for i, date in enumerate(dates_test):
            n_articles = np.random.randint(80, 200)
            for _ in range(n_articles):
                n_themes = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                selected_themes = np.random.choice(gdelt_themes, n_themes, replace=False)
                
                text_parts = []
                for theme in selected_themes:
                    processed_theme = theme.replace('_', ' ').lower()
                    text_parts.append(processed_theme)
                
                text = ' '.join(text_parts)
                test_data.append({'date': date, 'text': text})
            
            progress_bar.progress(0.7 + (i + 1) / len(dates_test) * 0.3)
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        
        # Enhanced display
        st.markdown("### 🎭 Demo Data Generated")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="success-card">
                <h4>🏋️ Training Data</h4>
                <p><strong>{len(train_df):,}</strong> records</p>
                <p>{len(dates_train)} days</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="success-card">
                <h4>🧪 Test Data</h4>
                <p><strong>{len(test_df):,}</strong> records</p>
                <p>{len(dates_test)} days</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Themes</h4>
                <p><strong>{len(gdelt_themes)}</strong> unique</p>
                <p>Multi-theme events</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👤 Generated By</h4>
                <p><strong>{CURRENT_USER}</strong></p>
                <p>{CURRENT_TIME}</p>
            </div>
            """, unsafe_allow_html=True)
        
        return train_df, test_df

class ProphetXGBoostTop3Forecaster:
    """Prophet + XGBoost Ensemble for Top 3 Hottest GDELT Topics - Enhanced Version"""
    
    def __init__(self, n_topics=10, top_k=3, forecast_horizon=7, batch_size=30000):
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
        
        # Enhanced GDELT stopwords
        self.gdelt_stopwords = {
            'wb', 'tax', 'fncact', 'soc', 'policy', 'pointsofinterest', 'crisislex', 
            'epu', 'uspec', 'ethnicity', 'worldlanguages', 'the', 'and', 'or', 
            'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'v2', 'sqldate', 'actor1', 'actor2', 'eventcode', 'goldsteinscale'
        }
        
        print(f"🔥 Enhanced Prophet + XGBoost Top-{top_k} GDELT Forecaster Initialized")
        print(f"   User: {CURRENT_USER} | Time: {CURRENT_TIME}")
    
    def memory_cleanup(self):
        """Enhanced memory cleanup"""
        gc.collect()
        if TF_AVAILABLE:
            try:
                tf.keras.backend.clear_session()
            except:
                pass
    
    def safe_preprocess_text(self, text):
        """Enhanced single text preprocessing"""
        try:
            if pd.isna(text) or text is None:
                return ""
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            words = [w for w in text.split() 
                    if len(w) > 2 and w not in self.gdelt_stopwords]
            return ' '.join(words[:50])  # Increased word limit
        except:
            return ""
    
    def batch_preprocess_fast(self, texts, batch_id=0):
        """Enhanced batch preprocessing with progress"""
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
        """Enhanced topic extraction with progress tracking"""
        st.write("⚡ **Extracting topics and identifying hot topics...**")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            # Step 1: TF-IDF Setup
            status_text.text("🎯 Setting up enhanced TF-IDF vectorizer...")
            progress_bar.progress(10)
            
            first_batch_texts = texts[:self.batch_size]
            first_batch_processed = self.batch_preprocess_fast(first_batch_texts, 0)
            
            if len(first_batch_processed) < 100:
                raise ValueError(f"Insufficient valid texts: {len(first_batch_processed)}")
            
            # Enhanced TF-IDF parameters
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased features
                ngram_range=(1, 2),
                min_df=max(3, len(first_batch_processed) // 1500),
                max_df=0.95,
                stop_words='english',
                lowercase=True,
                token_pattern=r'[a-zA-Z]{3,}'  # Minimum 3 characters
            )
            
            progress_bar.progress(20)
            
            # Step 2: Enhanced LDA Training
            status_text.text("🔄 Training enhanced LDA model...")
            first_tfidf = self.vectorizer.fit_transform(first_batch_processed)
            
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,  # Increased iterations
                learning_method='batch',
                batch_size=2048,  # Increased batch size
                n_jobs=1,
                verbose=0,
                doc_topic_prior=0.1,  # Alpha parameter
                topic_word_prior=0.01  # Beta parameter
            )
            
            first_topic_dist = self.lda_model.fit_transform(first_tfidf)
            progress_bar.progress(40)
            
            # Enhanced topic display
            feature_names = self.vectorizer.get_feature_names_out()
            st.write("🎯 **Discovered Topics with Enhanced Analysis:**")
            
            topic_display = st.empty()
            with topic_display.container():
                for i, topic in enumerate(self.lda_model.components_):
                    top_words = [feature_names[j] for j in topic.argsort()[-8:][::-1]]  # More words
                    self.topic_words[i] = top_words
                    
                    # Calculate topic coherence score
                    coherence_score = np.mean(topic.argsort()[-8:][::-1])
                    
                    st.write(f"   **Topic {i:2d}:** {', '.join(top_words[:5])} | Score: {coherence_score:.1f}")
            
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
            
            # Step 5: Enhanced hot topic identification
            status_text.text("🔥 Identifying hot topics with advanced metrics...")
            progress_bar.progress(95)
            
            self.identify_hot_topics(combined_topic_dist, dates)
            
            progress_bar.progress(100)
            status_text.text("✅ Enhanced topic extraction completed!")
            
            return combined_topic_dist
            
        except Exception as e:
            st.error(f"❌ Topic extraction failed: {e}")
            
            # Enhanced error handling
            with st.expander("🔍 Debug Information", expanded=False):
                import traceback
                st.code(traceback.format_exc())
                st.write(f"User: {CURRENT_USER}, Time: {CURRENT_TIME}")
            
            return np.random.dirichlet(np.ones(self.n_topics), len(texts))
    
    def identify_hot_topics(self, topic_dist, dates):
        """Enhanced hot topic identification with comprehensive metrics"""
        df = pd.DataFrame(topic_dist, columns=[f'topic_{i}' for i in range(self.n_topics)])
        df['date'] = pd.to_datetime(dates)
        
        topic_scores = {}
        
        for topic_idx in range(self.n_topics):
            topic_col = f'topic_{topic_idx}'
            
            # Enhanced hotness metrics
            avg_prob = df[topic_col].mean()
            recent_cutoff = int(0.7 * len(df))
            recent_avg = df[topic_col].iloc[recent_cutoff:].mean()
            variance = df[topic_col].var()
            std_dev = df[topic_col].std()
            
            daily_avg = df.groupby('date')[topic_col].mean()
            peak_intensity = daily_avg.max()
            growth_trend = daily_avg.iloc[-5:].mean() - daily_avg.iloc[:5].mean()
            
            daily_max_topic = df.groupby('date').apply(
                lambda x: x[[f'topic_{i}' for i in range(self.n_topics)]].mean().idxmax()
            )
            dominance_freq = (daily_max_topic == topic_col).sum() / len(daily_max_topic)
            
            # New metrics
            consistency = 1 - (std_dev / avg_prob) if avg_prob > 0 else 0
            momentum = max(0, growth_trend)
            
            # Enhanced hotness score with more sophisticated weighting
            hotness_score = (
                0.25 * avg_prob +
                0.25 * recent_avg +
                0.15 * variance +
                0.10 * peak_intensity +
                0.10 * dominance_freq +
                0.10 * consistency +
                0.05 * momentum
            )
            
            topic_scores[topic_idx] = {
                'hotness_score': hotness_score,
                'avg_prob': avg_prob,
                'recent_avg': recent_avg,
                'variance': variance,
                'peak_intensity': peak_intensity,
                'dominance_freq': dominance_freq,
                'consistency': consistency,
                'momentum': momentum,
                'growth_trend': growth_trend
            }
        
        # Select top hot topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1]['hotness_score'], reverse=True)
        self.hot_topics = [topic_idx for topic_idx, _ in sorted_topics[:self.top_k]]
        self.topic_popularity = topic_scores
        
        # Enhanced topic display
        st.markdown(f"### 🏆 **Top {self.top_k} Hot Topics (Enhanced Analysis):**")
        
        for rank, topic_idx in enumerate(self.hot_topics, 1):
            scores = topic_scores[topic_idx]
            topic_words = self.topic_words.get(topic_idx, [])
            
            # Create enhanced visualization
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                <div class="hot-topic-card">
                    <h4>🔥 #{rank}. Topic {topic_idx}: {', '.join(topic_words[:3])}</h4>
                    <p><strong>🏷️ Keywords:</strong> {', '.join(topic_words)}</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div><strong>🔥 Hotness Score:</strong> {scores['hotness_score']:.4f}</div>
                        <div><strong>📈 Avg Probability:</strong> {scores['avg_prob']:.4f}</div>
                        <div><strong>🎯 Dominance:</strong> {scores['dominance_freq']:.2%}</div>
                        <div><strong>📊 Consistency:</strong> {scores['consistency']:.3f}</div>
                        <div><strong>🚀 Momentum:</strong> {scores['momentum']:.4f}</div>
                        <div><strong>⚡ Peak Intensity:</strong> {scores['peak_intensity']:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Mini chart for topic trend
                daily_avg = df.groupby('date')[f'topic_{topic_idx}'].mean()
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(
                    y=daily_avg.values,
                    mode='lines',
                    name=f'Topic {topic_idx}',
                    line=dict(color='#FF4B4B', width=2)
                ))
                fig_mini.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=20, b=0),
                    title=f"Topic {topic_idx} Trend",
                    showlegend=False
                )
                st.plotly_chart(fig_mini, use_container_width=True)
    
    def prepare_time_series_data(self, topic_dist, dates):
        """Enhanced time series preparation"""
        st.write("📊 **Preparing enhanced time series data...**")
        
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
            
            # Enhanced time features
            daily_data['day_of_week'] = daily_data['date'].dt.dayofweek
            daily_data['day_of_month'] = daily_data['date'].dt.day
            daily_data['month'] = daily_data['date'].dt.month
            daily_data['quarter'] = daily_data['date'].dt.quarter
            daily_data['is_weekend'] = daily_data['day_of_week'].isin([5, 6]).astype(int)
            daily_data['is_month_start'] = daily_data['date'].dt.is_month_start.astype(int)
            daily_data['is_month_end'] = daily_data['date'].dt.is_month_end.astype(int)
            
            progress_bar.progress(75)
            
            # Enhanced features for hot topics only
            for lag in [1, 2, 3, 7, 14]:  # More lag features
                for topic_idx in self.hot_topics:
                    daily_data[f'topic_{topic_idx}_lag_{lag}'] = daily_data[f'topic_{topic_idx}'].shift(lag)
            
            for window in [3, 7, 14]:  # More window sizes
                for topic_idx in self.hot_topics:
                    daily_data[f'topic_{topic_idx}_ma_{window}'] = daily_data[f'topic_{topic_idx}'].rolling(window).mean()
                    daily_data[f'topic_{topic_idx}_std_{window}'] = daily_data[f'topic_{topic_idx}'].rolling(window).std()
            
            # Cross-topic interactions for all hot topics
            for i, topic_i in enumerate(self.hot_topics):
                for j, topic_j in enumerate(self.hot_topics):
                    if i < j:
                        daily_data[f'topic_{topic_i}_x_{topic_j}'] = daily_data[f'topic_{topic_i}'] * daily_data[f'topic_{topic_j}']
                        daily_data[f'topic_{topic_i}_ratio_{topic_j}'] = daily_data[f'topic_{topic_i}'] / (daily_data[f'topic_{topic_j}'] + 1e-8)
            
            daily_data = daily_data.dropna().reset_index(drop=True)
            
            progress_bar.progress(100)
            
            st.success(f"✅ Enhanced time series data prepared: {len(daily_data)} days with {daily_data.shape[1]} features")
            
            # Show feature breakdown
            feature_types = {
                'Base Topics': len([col for col in daily_data.columns if col.startswith('topic_') and '_' not in col[6:]]),
                'Lag Features': len([col for col in daily_data.columns if '_lag_' in col]),
                'Moving Averages': len([col for col in daily_data.columns if '_ma_' in col]),
                'Standard Deviations': len([col for col in daily_data.columns if '_std_' in col]),
                'Interactions': len([col for col in daily_data.columns if '_x_' in col or '_ratio_' in col]),
                'Time Features': len([col for col in daily_data.columns if col in ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']])
            }
            
            with st.expander("🔍 Feature Engineering Summary", expanded=False):
                for feature_type, count in feature_types.items():
                    st.write(f"**{feature_type}:** {count} features")
            
            return daily_data
            
        except Exception as e:
            st.error(f"❌ Enhanced time series preparation failed: {e}")
            return None
    
    def train_ensemble_models(self, daily_data):
        """Enhanced ensemble model training"""
        st.write("🚀 **Training enhanced ensemble models...**")
        
        total_steps = 3 if self.use_lstm else 2
        current_step = 0
        
        # Enhanced Prophet training
        st.write("📈 Training enhanced Prophet models...")
        prophet_progress = st.progress(0)
        
        # Enhanced Prophet parameters
        prophet_params = {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'yearly_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,  # More sensitive to changes
            'seasonality_prior_scale': 15.0,
            'holidays_prior_scale': 15.0,
            'interval_width': 0.85,
            'mcmc_samples': 0,  # Faster training
            'n_changepoints': 25  # More changepoints
        }
        
        for i, topic_idx in enumerate(self.hot_topics):
            prophet_data = pd.DataFrame({
                'ds': daily_data['date'],
                'y': daily_data[f'topic_{topic_idx}']
            })
            
            model = Prophet(**prophet_params)
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_data)
            
            self.prophet_models[f'topic_{topic_idx}'] = model
            
            future = model.make_future_dataframe(periods=self.forecast_horizon)
            forecast = model.predict(future)
            self.prophet_forecasts[f'topic_{topic_idx}'] = forecast
            
            prophet_progress.progress((i + 1) / len(self.hot_topics))
        
        current_step += 1
        st.success(f"✅ Enhanced Prophet models trained ({len(self.prophet_models)} models)")
        
        # Enhanced XGBoost training
        st.write("🚀 Training enhanced XGBoost models...")
        xgb_progress = st.progress(0)
        
        time_features = ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
        lag_features = [col for col in daily_data.columns if 'lag_' in col or 'ma_' in col or 'std_' in col]
        interaction_features = [col for col in daily_data.columns if '_x_' in col or '_ratio_' in col]
        
        for i, topic_idx in enumerate(self.hot_topics):
            other_hot_topics = [f'topic_{j}' for j in self.hot_topics if j != topic_idx]
            X_features = time_features + lag_features + interaction_features + other_hot_topics
            
            X = daily_data[X_features].values
            y = daily_data[f'topic_{topic_idx}'].values
            
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Enhanced XGBoost parameters
            model = xgb.XGBRegressor(
                n_estimators=150,  # More trees
                max_depth=8,  # Deeper trees
                learning_rate=0.08,  # Slightly lower learning rate
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=0.1,  # L2 regularization
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=15,
                verbose=False
            )
            
            self.xgboost_models[f'topic_{topic_idx}'] = model
            
            # Store enhanced feature importance
            self.feature_importance[f'topic_{topic_idx}'] = dict(zip(X_features, model.feature_importances_))
            
            xgb_progress.progress((i + 1) / len(self.hot_topics))
        
        current_step += 1
        st.success(f"✅ Enhanced XGBoost models trained ({len(self.xgboost_models)} models)")
        
        # Enhanced LSTM training (optional)
        if self.use_lstm:
            st.write("🔄 Training enhanced LSTM model...")
            lstm_progress = st.progress(0)
            
            try:
                hot_topic_cols = [f'topic_{i}' for i in self.hot_topics]
                data = daily_data[hot_topic_cols].values
                
                scaled_data = self.scaler.fit_transform(data)
                
                sequence_length = 10  # Increased sequence length
                X, y = [], []
                
                for i in range(sequence_length, len(scaled_data)):
                    X.append(scaled_data[i-sequence_length:i])
                    y.append(scaled_data[i])
                
                X, y = np.array(X), np.array(y)
                
                if len(X) >= 15:  # Minimum samples required
                    split_idx = int(0.8 * len(X))
                    X_train, X_val = X[:split_idx], X[split_idx:]
                    y_train, y_val = y[:split_idx], y[split_idx:]
                    
                    # Enhanced LSTM architecture
                    model = Sequential([
                        LSTM(32, return_sequences=True, input_shape=(sequence_length, self.top_k)),
                        Dropout(0.3),
                        LSTM(16, return_sequences=False),
                        Dropout(0.3),
                        Dense(16, activation='relu'),
                        Dropout(0.2),
                        Dense(self.top_k, activation='linear')
                    ])
                    
                    model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss='mse',
                        metrics=['mae']
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=30,  # More epochs
                        batch_size=32,
                        verbose=0,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                        ]
                    )
                    
                    self.lstm_model = model
                    lstm_progress.progress(100)
                    st.success("✅ Enhanced LSTM model trained")
                else:
                    st.warning("⚠️ Insufficient data for LSTM, skipping...")
                    self.use_lstm = False
                    
            except Exception as e:
                st.warning(f"⚠️ Enhanced LSTM training failed: {e}")
                self.use_lstm = False
        
        return True

def emergency_recovery():
    """Enhanced emergency recovery when ZIP processing fails"""
    
    st.markdown("### 🚨 Emergency Recovery Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All Cache", type="secondary"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key not in ['current_user', 'start_time']:
                    del st.session_state[key]
            
            # Clear Streamlit cache
            st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            
            # Force garbage collection
            gc.collect()
            
            st.success("✅ All cache cleared!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("💾 Emergency Mode", type="secondary"):
            st.session_state['emergency_mode'] = True
            st.session_state['max_file_size'] = 50  # 50MB limit
            st.session_state['max_files'] = 5  # 5 files limit
            st.info("✅ Emergency mode enabled - strict limits applied")
            st.rerun()
    
    with col3:
        if st.button("🏥 System Status", type="secondary"):
            memory = psutil.virtual_memory()
            st.write(f"💾 **Memory:** {memory.percent:.1f}% used")
            st.write(f"💻 **Available:** {memory.available / (1024**3):.1f} GB")
            st.write(f"🖥️ **CPU Cores:** {psutil.cpu_count()}")
            st.write(f"👤 **User:** {CURRENT_USER}")
            st.write(f"🕐 **Time:** {CURRENT_TIME}")

def init_session_state():
    """Initialize all session state variables with current info"""
    defaults = {
        'step': 1,
        'zip_structure': None,
        'zip_file': None,
        'selected_train_files': None,
        'selected_test_files': None,
        'train_data': None,
        'test_data': None,
        'model_trained': False,
        'forecaster': None,
        'predictions': None,
        'actuals': None,
        'results': None,
        'current_user': CURRENT_USER,
        'start_time': CURRENT_TIME,
        'emergency_mode': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """Enhanced main application function"""
    init_session_state()
    
    # Enhanced header with user info
    st.markdown('<h1 class="main-header">🔥 GDELT Hot Topics Forecaster</h1>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="user-info">
        👤 <strong>User:</strong> {CURRENT_USER} | 
        🕐 <strong>Session:</strong> {CURRENT_TIME} UTC | 
        🔥 <strong>Enhanced Pipeline:</strong> ZIP Upload → Advanced Processing → AI Forecasting
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced progress indicator
    steps = ["📁 Upload ZIP", "🔍 Explore Data", "📊 Process Data", "🔥 Train Models", "📈 View Results"]
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
    
    # STEP 1: Enhanced Upload ZIP file
    if st.session_state.step == 1:
        st.markdown('<div class="step-container"><h2>📁 STEP 1: Enhanced GDELT Data Upload</h2></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📤 Upload ZIP File (Optimized Engine)")
            
            # Enhanced file requirements
            emergency_limits = st.session_state.get('emergency_mode', False)
            max_size = 50 if emergency_limits else 150
            max_files = 5 if emergency_limits else 15
            
            st.markdown(f"""
            <div class="{'warning-card' if emergency_limits else 'file-info-card'}">
                <h4>⚠️ File Requirements {'(Emergency Mode)' if emergency_limits else '(Optimized)'}</h4>
                <ul>
                    <li><strong>Maximum ZIP size:</strong> {max_size}MB</li>
                    <li><strong>Maximum CSV files:</strong> {max_files}</li>
                    <li><strong>Individual file limit:</strong> 30MB</li>
                    <li><strong>Supported encodings:</strong> UTF-8, Latin-1, CP1252</li>
                    <li><strong>Required columns:</strong> DATE, THEMES (or similar)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_zip = st.file_uploader(
                "Upload GDELT data ZIP file containing CSV files",
                type=['zip'],
                help="Upload ZIP file with GDELT CSV data organized by month (April-May for training, June for testing)"
            )
            
            if uploaded_zip is not None:
                # Use optimized processor
                processor = OptimizedGDELTDataProcessor()
                
                if emergency_limits:
                    processor.max_file_size_mb = 50
                    processor.max_files_to_process = 5
                
                with st.spinner("🔍 Safely analyzing ZIP file with enhanced engine..."):
                    zip_structure = processor.safe_explore_zip_structure(uploaded_zip)
                
                if zip_structure:
                    st.session_state.zip_structure = zip_structure
                    st.session_state.zip_file = uploaded_zip
                    
                    st.success("✅ ZIP file analyzed successfully with enhanced engine!")
                    
                    # Enhanced analysis summary
                    with st.expander("📊 Detailed Analysis Summary", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 CSV Files", len(zip_structure['csv_files']))
                            st.metric("📁 Total Size", f"{zip_structure['file_size_mb']:.1f} MB")
                        
                        with col2:
                            st.metric("🏋️ Training Files", len(zip_structure['file_analysis']['train_candidates']))
                            st.metric("🧪 Test Files", len(zip_structure['file_analysis']['test_candidates']))
                        
                        with col3:
                            st.metric("❓ Unknown Files", len(zip_structure['file_analysis']['unknown_files']))
                            st.metric("⚠️ Skipped Files", len(zip_structure['skipped_files']))
                        
                        with col4:
                            stats = zip_structure['processing_stats']
                            st.metric("⚙️ Engine Version", "Enhanced v2.0")
                            st.metric("👤 Processed By", CURRENT_USER)
                    
                    if st.button("🚀 Continue to Enhanced Data Exploration", type="primary"):
                        st.session_state.step = 2
                        st.rerun()
                else:
                    st.error("❌ Failed to analyze ZIP file with enhanced engine.")
                    
                    # Show emergency options
                    st.markdown("### 🚨 Recovery Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔧 Try Emergency Mode"):
                            st.session_state.emergency_mode = True
                            st.rerun()
                    
                    with col2:
                        if st.button("🎭 Use Demo Data Instead"):
                            processor = OptimizedGDELTDataProcessor()
                            train_data, test_data = processor.create_demo_data()
                            
                            st.session_state.train_data = train_data
                            st.session_state.test_data = test_data
                            st.session_state.step = 4
                            st.rerun()
        
        with col2:
            st.markdown("### 🎭 Quick Start Options")
            
            # Enhanced demo option
            st.markdown("""
            <div class="success-card">
                <h4>🎭 Demo Data Available</h4>
                <p>Use realistic GDELT demo data to explore all features without uploading files.</p>
                <ul>
                    <li>📊 15,000+ demo records</li>
                    <li>🏋️ 2 months training data</li>
                    <li>🧪 10 days test data</li>
                    <li>🔥 15 diverse topics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🎭 Use Enhanced Demo Data", type="secondary"):
                processor = OptimizedGDELTDataProcessor()
                train_data, test_data = processor.create_demo_data()
                
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                st.session_state.step = 4
                st.success("✅ Enhanced demo data loaded! Skipping to model training...")
                time.sleep(1)
                st.rerun()
            
            # System status
            st.markdown("### 📊 System Status")
            memory = psutil.virtual_memory()
            
            if memory.percent > 80:
                st.error(f"⚠️ High memory usage: {memory.percent:.1f}%")
            elif memory.percent > 60:
                st.warning(f"⚠️ Moderate memory usage: {memory.percent:.1f}%")
            else:
                st.success(f"✅ Memory OK: {memory.percent:.1f}%")
            
            st.info(f"💾 Available: {memory.available / (1024**3):.1f} GB")
            st.info(f"🖥️ CPU Cores: {psutil.cpu_count()}")
    
    # STEP 2: Enhanced Explore data structure
    elif st.session_state.step == 2:
        st.markdown('<div class="step-container"><h2>🔍 STEP 2: Enhanced Data Structure Explorer</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.zip_structure:
            zip_structure = st.session_state.zip_structure
            file_analysis = zip_structure['file_analysis']
            
            # Enhanced file categorization display
            st.markdown("### 📂 Intelligent File Categorization")
            
            tab1, tab2, tab3 = st.tabs(["🏋️ Training Data", "🧪 Test Data", "❓ Unknown Files"])
            
            with tab1:
                st.markdown("#### 🏋️ Training Data Candidates (April-May)")
                if file_analysis['train_candidates']:
                    for i, file in enumerate(file_analysis['train_candidates'], 1):
                        st.markdown(f"""
                        <div class="success-card">
                            <h5>📄 {i}. {os.path.basename(file)}</h5>
                            <p><strong>Path:</strong> <code>{file}</code></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No training candidates found automatically.")
                    st.info("💡 You can manually select files in the next step.")
            
            with tab2:
                st.markdown("#### 🧪 Test Data Candidates (June)")
                if file_analysis['test_candidates']:
                    for i, file in enumerate(file_analysis['test_candidates'], 1):
                        st.markdown(f"""
                        <div class="success-card">
                            <h5>📄 {i}. {os.path.basename(file)}</h5>
                            <p><strong>Path:</strong> <code>{file}</code></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No test candidates found automatically.")
                    st.info("💡 You can manually select files in the next step.")
            
            with tab3:
                st.markdown("#### ❓ Unknown Files")
                if file_analysis['unknown_files']:
                    for i, file in enumerate(file_analysis['unknown_files'][:10], 1):
                        st.write(f"📄 {i}. `{os.path.basename(file)}`")
                    if len(file_analysis['unknown_files']) > 10:
                        st.write(f"... and {len(file_analysis['unknown_files']) - 10} more files")
                else:
                    st.success("✅ All files successfully categorized!")
            
            # Enhanced file selection
            st.markdown("### 📋 Enhanced File Selection")
            
            all_csv_files = zip_structure['csv_files']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏋️ Select Training Files")
                selected_train_files = st.multiselect(
                    "Choose CSV files for training (April-May data):",
                    options=all_csv_files,
                    default=file_analysis['train_candidates'][:10],
                    help="Select CSV files containing April-May data for model training",
                    key="train_files_selector"
                )
                
                if selected_train_files:
                    st.success(f"✅ Selected {len(selected_train_files)} training files")
            
            with col2:
                st.markdown("#### 🧪 Select Test Files")
                selected_test_files = st.multiselect(
                    "Choose CSV files for testing (June data):",
                    options=all_csv_files,
                    default=file_analysis['test_candidates'][:5],
                    help="Select CSV files containing June data for model testing",
                    key="test_files_selector"
                )
                
                if selected_test_files:
                    st.success(f"✅ Selected {len(selected_test_files)} test files")
            
            # Enhanced validation and proceed
            if selected_train_files and selected_test_files:
                st.markdown("### ✅ Selection Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(selected_train_files)}</h3>
                        <p>🏋️ Training Files</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(selected_test_files)}</h3>
                        <p>🧪 Test Files</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_files = len(selected_train_files) + len(selected_test_files)
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{total_files}</h3>
                        <p>📊 Total Selected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.selected_train_files = selected_train_files
                st.session_state.selected_test_files = selected_test_files
                
                if st.button("📊 Process Selected Files with Enhanced Engine", type="primary"):
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.warning("⚠️ Please select both training and test files to continue")
                
                # Show selection hints
                if not selected_train_files:
                    st.info("💡 **Training files:** Look for files with 'april', 'may', '04', '05' in the name")
                if not selected_test_files:
                    st.info("💡 **Test files:** Look for files with 'june', 'jun', '06' in the name")
    
    # STEP 3: Enhanced Process data
    elif st.session_state.step == 3:
        st.markdown('<div class="step-container"><h2>📊 STEP 3: Enhanced GDELT Data Processing</h2></div>', unsafe_allow_html=True)
        
        processor = OptimizedGDELTDataProcessor()
        
        # Enhanced processing configuration
        st.markdown("### ⚙️ Enhanced Processing Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_records_per_file = st.slider(
                "Max records per file", 
                5000, 50000, 25000, step=5000,
                help="Limit records per file to prevent memory issues"
            )
        
        with col2:
            enable_sampling = st.checkbox(
                "Enable intelligent sampling", 
                value=True,
                help="Use smart sampling for large files"
            )
        
        with col3:
            parallel_processing = st.checkbox(
                "Enable parallel processing",
                value=False,
                help="Process multiple files simultaneously (experimental)"
            )
        
        # Enhanced processing summary
        st.markdown(f"""
        <div class="file-info-card">
            <h4>🔧 Processing Configuration</h4>
            <p><strong>Max Records:</strong> {max_records_per_file:,} per file</p>
            <p><strong>Smart Sampling:</strong> {'✅ Enabled' if enable_sampling else '❌ Disabled'}</p>
            <p><strong>Parallel Processing:</strong> {'✅ Enabled' if parallel_processing else '❌ Disabled'}</p>
            <p><strong>User:</strong> {CURRENT_USER} | <strong>Time:</strong> {CURRENT_TIME}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Process training files with enhanced engine
        st.markdown("### 🏋️ Processing Training Data (Enhanced Engine)")
        train_dataframes = []
        
        if st.session_state.selected_train_files:
            
            train_progress = st.progress(0)
            train_status = st.empty()
            train_container = st.container()
            
            for i, train_file in enumerate(st.session_state.selected_train_files):
                train_status.text(f"🔄 Processing {os.path.basename(train_file)} ({i+1}/{len(st.session_state.selected_train_files)})...")
                
                with train_container.expander(f"📄 {os.path.basename(train_file)}", expanded=False):
                    # Safe extraction with enhanced error handling
                    df, separator, encoding = processor.safe_extract_and_read_csv(
                        st.session_state.zip_file, 
                        train_file,
                        max_rows=max_records_per_file if enable_sampling else None
                    )
                    
                    if df is not None:
                        # Safe processing with enhanced features
                        processed_df = processor.safe_process_gdelt_dataframe(
                            df, 
                            f"train_{i+1}",
                            max_records=max_records_per_file if enable_sampling else None
                        )
                        
                        if processed_df is not None:
                            train_dataframes.append(processed_df)
                            st.info(f"✅ Successfully processed {len(processed_df):,} records")
                            
                            # Enhanced memory cleanup
                            del df, processed_df
                            gc.collect()
                        else:
                            st.error(f"❌ Failed to process {train_file}")
                    else:
                        st.error(f"❌ Failed to read {train_file}")
                
                train_progress.progress((i + 1) / len(st.session_state.selected_train_files))
            
            train_status.text("✅ Training data processing completed!")
        
        # Process test files with enhanced engine
        st.markdown("### 🧪 Processing Test Data (Enhanced Engine)")
        test_dataframes = []
        
        if st.session_state.selected_test_files:
            
            test_progress = st.progress(0)
            test_status = st.empty()
            test_container = st.container()
            
            for i, test_file in enumerate(st.session_state.selected_test_files):
                test_status.text(f"🔄 Processing {os.path.basename(test_file)} ({i+1}/{len(st.session_state.selected_test_files)})...")
                
                with test_container.expander(f"📄 {os.path.basename(test_file)}", expanded=False):
                    # Safe extraction with enhanced error handling
                    df, separator, encoding = processor.safe_extract_and_read_csv(
                        st.session_state.zip_file, 
                        test_file,
                        max_rows=max_records_per_file if enable_sampling else None
                    )
                    
                    if df is not None:
                        # Safe processing with enhanced features
                        processed_df = processor.safe_process_gdelt_dataframe(
                            df, 
                            f"test_{i+1}",
                            max_records=max_records_per_file if enable_sampling else None
                        )
                        
                        if processed_df is not None:
                            test_dataframes.append(processed_df)
                            st.info(f"✅ Successfully processed {len(processed_df):,} records")
                            
                            # Enhanced memory cleanup
                            del df, processed_df
                            gc.collect()
                        else:
                            st.error(f"❌ Failed to process {test_file}")
                    else:
                        st.error(f"❌ Failed to read {test_file}")
                
                test_progress.progress((i + 1) / len(st.session_state.selected_test_files))
            
            test_status.text("✅ Test data processing completed!")
        
        # Enhanced data combination and finalization
        if train_dataframes and test_dataframes:
            st.markdown("### 🔗 Enhanced Data Combination & Finalization")
            
            combine_progress = st.progress(0)
            combine_status = st.empty()
            
            combine_status.text("🔗 Combining training datasets...")
            combine_progress.progress(20)
            
            train_data = pd.concat(train_dataframes, ignore_index=True)
            train_data = train_data.sort_values('date').reset_index(drop=True)
            train_data = train_data.drop_duplicates().reset_index(drop=True)  # Remove duplicates
            
            combine_status.text("🔗 Combining test datasets...")
            combine_progress.progress(40)
            
            test_data = pd.concat(test_dataframes, ignore_index=True)
            test_data = test_data.sort_values('date').reset_index(drop=True)
            test_data = test_data.drop_duplicates().reset_index(drop=True)  # Remove duplicates
            
            combine_status.text("✂️ Optimizing test data for forecasting...")
            combine_progress.progress(60)
            
            # Enhanced test data optimization
            unique_test_dates = sorted(test_data['date'].dt.date.unique())
            if len(unique_test_dates) > 10:
                first_10_dates = unique_test_dates[:10]
                test_data = test_data[test_data['date'].dt.date.isin(first_10_dates)].copy()
                st.info(f"📅 Optimized test data to first 10 days: {first_10_dates[0]} to {first_10_dates[-1]}")
            
            combine_status.text("🧹 Final data cleaning and optimization...")
            combine_progress.progress(80)
            
            # Enhanced data quality checks
            initial_train_count = len(train_data)
            initial_test_count = len(test_data)
            
            # Remove very short texts
            train_data = train_data[train_data['text'].str.len() >= 10]
            test_data = test_data[test_data['text'].str.len() >= 10]
            
            # Remove texts with too few words
            train_data = train_data[train_data['text'].str.split().str.len() >= 3]
            test_data = test_data[test_data['text'].str.split().str.len() >= 3]
            
            combine_progress.progress(100)
            combine_status.text("✅ Enhanced data combination completed!")
            
            # Enhanced final dataset summary
            st.markdown("### 📊 Enhanced Dataset Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏋️ Training Dataset")
                
                col1a, col1b = st.columns(2)
                
                with col1a:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{len(train_data):,}</h3>
                        <p>📊 Total Records</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col1b:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{train_data['date'].nunique()}</h3>
                        <p>📅 Unique Days</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                train_quality = (len(train_data) / initial_train_count) * 100
                st.info(f"📅 **Period:** {train_data['date'].min().date()} → {train_data['date'].max().date()}")
                st.info(f"🎯 **Data Quality:** {train_quality:.1f}% retained after cleaning")
            
            with col2:
                st.markdown("#### 🧪 Test Dataset")
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{len(test_data):,}</h3>
                        <p>📊 Total Records</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2b:
                    st.markdown(f"""
                    <div class="success-card">
                        <h3>{test_data['date'].nunique()}</h3>
                        <p>📅 Unique Days</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                test_quality = (len(test_data) / initial_test_count) * 100
                st.info(f"📅 **Period:** {test_data['date'].min().date()} → {test_data['date'].max().date()}")
                st.info(f"🎯 **Data Quality:** {test_quality:.1f}% retained after cleaning")
            
            # Enhanced data statistics
            st.markdown("### 📈 Enhanced Data Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_daily_train = train_data.groupby(train_data['date'].dt.date).size().mean()
                st.metric("📈 Avg Daily (Train)", f"{avg_daily_train:.1f}")
            
            with col2:
                avg_daily_test = test_data.groupby(test_data['date'].dt.date).size().mean()
                st.metric("📈 Avg Daily (Test)", f"{avg_daily_test:.1f}")
            
            with col3:
                avg_text_len_train = train_data['text'].str.len().mean()
                st.metric("📝 Avg Text Len", f"{avg_text_len_train:.0f}")
            
            with col4:
                total_processing_time = time.time()
                st.metric("⏱️ Processing", f"{len(st.session_state.selected_train_files) + len(st.session_state.selected_test_files)} files")
            
            # Enhanced data preview
            st.markdown("### 👀 Enhanced Data Preview")
            
            tab1, tab2 = st.tabs(["🏋️ Training Sample", "🧪 Test Sample"])
            
            with tab1:
                st.markdown("#### Recent Training Data")
                st.dataframe(train_data.tail(10), use_container_width=True)
            
            with tab2:
                st.markdown("#### Recent Test Data")
                st.dataframe(test_data.tail(10), use_container_width=True)
            
            # Store processed data with metadata
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.session_state.processing_metadata = {
                'train_files_processed': len(st.session_state.selected_train_files),
                'test_files_processed': len(st.session_state.selected_test_files),
                'train_records': len(train_data),
                'test_records': len(test_data),
                'processing_time': CURRENT_TIME,
                'user': CURRENT_USER,
                'quality_train': train_quality,
                'quality_test': test_quality
            }
            
            st.success("✅ Enhanced data processing completed successfully!")
            
            if st.button("🔥 Continue to Enhanced Model Training", type="primary"):
                st.session_state.step = 4
                st.rerun()
        
        else:
            st.error("❌ Failed to process data files. Please check your file selection and try again.")
            
            # Enhanced error recovery options
            st.markdown("### 🚨 Recovery Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔙 Go Back to File Selection"):
                    st.session_state.step = 2
                    st.rerun()
            
            with col2:
                if st.button("🎭 Use Demo Data Instead"):
                    processor = OptimizedGDELTDataProcessor()
                    train_data, test_data = processor.create_demo_data()
                    
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.session_state.step = 4
                    st.rerun()
            
            with col3:
                if st.button("🔧 Enable Emergency Mode"):
                    st.session_state.emergency_mode = True
                    st.session_state.step = 1
                    st.rerun()
    
    # STEP 4: Enhanced Model training
    elif st.session_state.step == 4:
        st.markdown('<div class="step-container"><h2>🔥 STEP 4: Enhanced Hot Topics AI Model Training</h2></div>', unsafe_allow_html=True)
        
        # Enhanced configuration section
        with st.sidebar:
            st.markdown("## ⚙️ Enhanced Model Configuration")
            st.markdown(f"👤 **User:** {CURRENT_USER}")
            st.markdown(f"🕐 **Time:** {CURRENT_TIME}")
            
            n_topics = st.slider("📊 Total Topics to Discover", 8, 25, 12, 
                                help="Number of topics to extract from text data")
            top_k = st.slider("🔥 Hot Topics to Focus", 2, 6, 3,
                             help="Number of hottest topics to focus on for forecasting")
            forecast_horizon = st.slider("📅 Forecast Horizon (days)", 3, 21, 7,
                                       help="Number of days to forecast ahead")
            batch_size = st.selectbox("⚡ Processing Batch Size", [20000, 30000, 40000], index=1,
                                    help="Batch size for processing large datasets")
            
            st.markdown("### 🎛️ Enhanced Ensemble Configuration")
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
            
            st.markdown("### 💾 Enhanced System Info")
            memory = psutil.virtual_memory()
            
            if memory.percent > 80:
                st.error(f"⚠️ High Memory: {memory.percent:.1f}%")
            else:
                st.success(f"✅ Memory OK: {memory.percent:.1f}%")
            
            st.metric("💻 CPU Cores", f"{os.cpu_count()}")
            st.metric("🤖 TensorFlow", "✅ Available" if TF_AVAILABLE else "❌ Not Available")
        
        # Enhanced main training interface
        if st.session_state.train_data is not None and st.session_state.test_data is not None:
            # Enhanced data summary
            st.markdown("### 📊 Enhanced Training Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{len(st.session_state.train_data):,}</h3>
                    <p>🏋️ Training Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{len(st.session_state.test_data):,}</h3>
                    <p>🧪 Test Records</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{st.session_state.train_data['date'].nunique()}</h3>
                    <p>📅 Training Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{st.session_state.test_data['date'].nunique()}</h3>
                    <p>📅 Test Days</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Enhanced training configuration summary
            st.markdown("### 🎯 Enhanced Training Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="file-info-card">
                    <h4>🔧 Model Parameters</h4>
                    <ul>
                        <li><strong>Total Topics:</strong> {n_topics}</li>
                        <li><strong>Hot Topics Focus:</strong> {top_k}</li>
                        <li><strong>Forecast Horizon:</strong> {forecast_horizon} days</li>
                        <li><strong>Batch Size:</strong> {batch_size:,}</li>
                        <li><strong>Engine Version:</strong> Enhanced v2.0</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="file-info-card">
                    <h4>⚖️ Ensemble Weights</h4>
                    <ul>
                        <li><strong>Prophet:</strong> {ensemble_prophet:.2f}</li>
                        <li><strong>XGBoost:</strong> {ensemble_xgboost:.2f}</li>
                        <li><strong>LSTM:</strong> {ensemble_lstm:.2f}</li>
                        <li><strong>User:</strong> {CURRENT_USER}</li>
                        <li><strong>Time:</strong> {CURRENT_TIME}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced training button and process
            if st.button("🚀 Start Enhanced Hot Topics AI Training", type="primary"):
                # Initialize enhanced forecaster
                forecaster = ProphetXGBoostTop3Forecaster(
                    n_topics=n_topics,
                    top_k=top_k,
                    forecast_horizon=forecast_horizon,
                    batch_size=batch_size
                )
                
                # Set enhanced ensemble weights
                forecaster.ensemble_weights = {
                    'prophet': ensemble_prophet,
                    'xgboost': ensemble_xgboost,
                    'lstm': ensemble_lstm
                }
                
                try:
                    st.markdown("### 🎯 Enhanced Training Progress")
                    training_start_time = time.time()
                    
                    # Enhanced topic extraction
                    st.markdown("#### 1️⃣ Enhanced Topic Extraction & Hot Topic Identification")
                    topic_dist = forecaster.extract_topics_and_identify_hot_topics(
                        st.session_state.train_data['text'].tolist(),
                        st.session_state.train_data['date'].tolist()
                    )
                    
                    # Enhanced time series preparation
                    st.markdown("#### 2️⃣ Enhanced Time Series Data Preparation")
                    daily_data = forecaster.prepare_time_series_data(
                        topic_dist, 
                        st.session_state.train_data['date'].tolist()
                    )
                    
                    if daily_data is None:
                        st.error("❌ Failed to prepare enhanced time series data")
                        return
                    
                    # Enhanced model training
                    st.markdown("#### 3️⃣ Enhanced Ensemble Model Training")
                    training_success = forecaster.train_ensemble_models(daily_data)
                    
                    if not training_success:
                        st.error("❌ Enhanced model training failed")
                        return
                    
                    # Enhanced forecasting
                    st.markdown("#### 4️⃣ Enhanced Forecast Generation")
                    
                    forecast_progress = st.progress(0)
                    forecast_status = st.empty()
                    
                    # Process test data with enhanced engine
                    forecast_status.text("📊 Processing test data with enhanced engine...")
                    forecast_progress.progress(15)
                    
                    test_topic_dist = []
                    test_texts = st.session_state.test_data['text'].tolist()
                    test_batch_size = min(15000, len(test_texts))  # Optimized batch size
                    
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
                        st.error("❌ Failed to process test data with enhanced engine")
                        return
                    
                    forecast_progress.progress(35)
                    
                    # Prepare enhanced test time series
                    forecast_status.text("📈 Preparing enhanced test time series...")
                    test_daily_data = forecaster.prepare_time_series_data(
                        test_topic_dist, 
                        st.session_state.test_data['date'].tolist()
                    )
                    
                    if test_daily_data is None:
                        st.error("❌ Failed to prepare enhanced test time series")
                        return
                    
                    forecast_progress.progress(55)
                    
                    # Generate enhanced predictions
                    forecast_status.text("🔮 Generating enhanced ensemble predictions...")
                    
                    # Enhanced Prophet predictions
                    prophet_preds = []
                    for topic_idx in forecaster.hot_topics:
                        model = forecaster.prophet_models[f'topic_{topic_idx}']
                        future_df = pd.DataFrame({'ds': test_daily_data['date']})
                        forecast = model.predict(future_df)
                        prophet_preds.append(forecast['yhat'].values)
                    
                    prophet_predictions = np.array(prophet_preds).T
                    
                    forecast_progress.progress(75)
                    
                    # Enhanced XGBoost predictions
                    xgb_predictions = []
                    time_features = ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend', 'is_month_start', 'is_month_end']
                    
                    for topic_idx in forecaster.hot_topics:
                        model = forecaster.xgboost_models[f'topic_{topic_idx}']
                        
                        # Prepare features for XGBoost
                        feature_cols = []
                        for col in test_daily_data.columns:
                            if any(feat in col for feat in ['lag_', 'ma_', 'std_', '_x_', '_ratio_']) or col in time_features:
                                feature_cols.append(col)
                        
                        # Add other hot topics as features
                        for other_topic in forecaster.hot_topics:
                            if other_topic != topic_idx:
                                feature_cols.append(f'topic_{other_topic}')
                        
                        # Get available features
                        available_features = [col for col in feature_cols if col in test_daily_data.columns]
                        
                        if available_features:
                            X_test = test_daily_data[available_features].fillna(0).values
                            pred = model.predict(X_test)
                            xgb_predictions.append(pred)
                        else:
                            # Fallback to Prophet if no features available
                            xgb_predictions.append(prophet_predictions[:, len(xgb_predictions)])
                    
                    xgb_predictions = np.array(xgb_predictions).T
                    
                    forecast_progress.progress(85)
                    
                    # Enhanced LSTM predictions (if available)
                    lstm_predictions = None
                    if forecaster.use_lstm and forecaster.lstm_model:
                        try:
                            # Prepare LSTM input
                            hot_topic_cols = [f'topic_{i}' for i in forecaster.hot_topics]
                            lstm_data = test_daily_data[hot_topic_cols].fillna(0).values
                            scaled_lstm_data = forecaster.scaler.transform(lstm_data)
                            
                            # Create sequences
                            sequence_length = 10
                            if len(scaled_lstm_data) >= sequence_length:
                                X_lstm = []
                                for i in range(sequence_length, len(scaled_lstm_data)):
                                    X_lstm.append(scaled_lstm_data[i-sequence_length:i])
                                
                                if X_lstm:
                                    X_lstm = np.array(X_lstm)
                                    lstm_pred = forecaster.lstm_model.predict(X_lstm, verbose=0)
                                    lstm_pred_rescaled = forecaster.scaler.inverse_transform(lstm_pred)
                                    
                                    # Pad to match other predictions
                                    padding = np.zeros((sequence_length, len(forecaster.hot_topics)))
                                    lstm_predictions = np.vstack([padding, lstm_pred_rescaled])
                        except Exception as e:
                            st.warning(f"⚠️ LSTM prediction failed: {e}")
                            lstm_predictions = None
                    
                    # Enhanced ensemble combination
                    final_predictions = (
                        ensemble_prophet * prophet_predictions +
                        ensemble_xgboost * xgb_predictions
                    )
                    
                    if lstm_predictions is not None:
                        # Ensure same shape
                        min_len = min(len(final_predictions), len(lstm_predictions))
                        final_predictions = final_predictions[:min_len]
                        lstm_predictions = lstm_predictions[:min_len]
                        final_predictions += ensemble_lstm * lstm_predictions
                    
                    forecast_progress.progress(95)
                    
                    # Get enhanced actual values
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
                    forecast_status.text("✅ Enhanced forecasting completed!")
                    
                    # Calculate enhanced metrics
                    mae = np.mean(np.abs(final_predictions - actual_values))
                    mse = np.mean((final_predictions - actual_values)**2)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((actual_values - final_predictions) / (actual_values + 1e-8))) * 100
                    
                    # Per-topic enhanced metrics
                    hot_topics_results = []
                    for i, topic_idx in enumerate(forecaster.hot_topics):
                        topic_mae = np.mean(np.abs(final_predictions[:, i] - actual_values[:, i]))
                        topic_mse = np.mean((final_predictions[:, i] - actual_values[:, i])**2)
                        topic_rmse = np.sqrt(topic_mse)
                        topic_mape = np.mean(np.abs((actual_values[:, i] - final_predictions[:, i]) / (actual_values[:, i] + 1e-8))) * 100
                        
                        hot_topics_results.append({
                            'topic': topic_idx,
                            'mae': topic_mae,
                            'mse': topic_mse,
                            'rmse': topic_rmse,
                            'mape': topic_mape,
                            'hotness_score': forecaster.topic_popularity[topic_idx]['hotness_score'],
                            'avg_prob': forecaster.topic_popularity[topic_idx]['avg_prob'],
                            'keywords': forecaster.topic_words.get(topic_idx, [])
                        })
                    
                    training_end_time = time.time()
                    training_duration = training_end_time - training_start_time
                    
                    # Store enhanced results
                    st.session_state.forecaster = forecaster
                    st.session_state.predictions = final_predictions
                    st.session_state.actuals = actual_values
                    st.session_state.results = {
                        'overall': {
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'mape': mape
                        },
                        'hot_topics': hot_topics_results,
                        'hot_topic_indices': forecaster.hot_topics,
                        'config': {
                            'n_topics': n_topics,
                            'top_k': top_k,
                            'forecast_horizon': forecast_horizon,
                            'batch_size': batch_size,
                            'ensemble_weights': {
                                'prophet': ensemble_prophet,
                                'xgboost': ensemble_xgboost,
                                'lstm': ensemble_lstm
                            },
                            'training_duration': training_duration,
                            'engine_version': 'Enhanced v2.0'
                        },
                        'test_dates': test_daily_actual['date'].tolist(),
                        'metadata': {
                            'user': CURRENT_USER,
                            'timestamp': CURRENT_TIME,
                            'train_records': len(st.session_state.train_data),
                            'test_records': len(st.session_state.test_data)
                        }
                    }
                    
                    st.session_state.model_trained = True
                    
                    st.success("🎉 **Enhanced Training and Forecasting Completed Successfully!**")
                    
                    # Enhanced quick results preview
                    st.markdown("### 🎯 Enhanced Performance Preview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>{mae:.4f}</h3>
                            <p>📈 Overall MAE</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>{rmse:.4f}</h3>
                            <p>📊 Overall RMSE</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>{mape:.1f}%</h3>
                            <p>🎯 Overall MAPE</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="success-card">
                            <h3>{training_duration:.1f}s</h3>
                            <p>⏱️ Training Time</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show best performing topic
                    best_topic = min(hot_topics_results, key=lambda x: x['mae'])
                    st.success(f"🏆 **Best Topic:** Topic {best_topic['topic']} (MAE: {best_topic['mae']:.4f}) - Keywords: {', '.join(best_topic['keywords'][:3])}")
                    
                    if st.button("📊 View Enhanced Detailed Results", type="primary"):
                        st.session_state.step = 5
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Enhanced training failed: {str(e)}")
                    
                    # Enhanced error information
                    with st.expander("🔍 Enhanced Debug Information", expanded=False):
                        import traceback
                        st.code(traceback.format_exc())
                        st.write(f"**User:** {CURRENT_USER}")
                        st.write(f"**Time:** {CURRENT_TIME}")
                        st.write(f"**Memory:** {psutil.virtual_memory().percent:.1f}%")
        
        else:
            st.error("❌ No data available for enhanced training. Please complete data processing first.")
            
            # Enhanced recovery options
            st.markdown("### 🚨 Enhanced Recovery Options")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔙 Back to Data Processing"):
                    st.session_state.step = 3
                    st.rerun()
            
            with col2:
                if st.button("🎭 Use Enhanced Demo Data"):
                    processor = OptimizedGDELTDataProcessor()
                    train_data, test_data = processor.create_demo_data()
                    
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.success("✅ Enhanced demo data loaded!")
                    st.rerun()
            
            with col3:
                if st.button("🔄 Reset to Beginning"):
                    for key in list(st.session_state.keys()):
                        if key not in ['current_user', 'start_time']:
                            del st.session_state[key]
                    st.session_state.step = 1
                    st.rerun()
    
    # STEP 5: Enhanced Results and visualization
    elif st.session_state.step == 5:
        st.markdown('<div class="step-container"><h2>📈 STEP 5: Enhanced Results & Comprehensive AI Analysis</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Enhanced header metrics
            st.markdown("### 🎯 Enhanced Performance Overview")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{results['overall']['mae']:.4f}</h3>
                    <p>📈 Overall MAE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{results['overall']['rmse']:.4f}</h3>
                    <p>📊 Overall RMSE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="success-card">
                    <h3>{results['overall']['mape']:.1f}%</h3>
                    <p>🎯 Overall MAPE</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_hotness = np.mean([t['hotness_score'] for t in results['hot_topics']])
                st.markdown(f"""
                <div class="success-card">
                    <h3>{avg_hotness:.3f}</h3>
                    <p>🔥 Avg Hotness</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                best_mae = min([t['mae'] for t in results['hot_topics']])
                st.markdown(f"""
                <div class="success-card">
                    <h3>{best_mae:.4f}</h3>
                    <p>🏆 Best Topic MAE</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced metadata display
            if 'metadata' in results:
                metadata = results['metadata']
                st.markdown(f"""
                <div class="file-info-card">
                    <h4>📊 Enhanced Analysis Metadata</h4>
                    <p><strong>User:</strong> {metadata['user']} | <strong>Generated:</strong> {metadata['timestamp']}</p>
                    <p><strong>Training Records:</strong> {metadata['train_records']:,} | <strong>Test Records:</strong> {metadata['test_records']:,}</p>
                    <p><strong>Engine Version:</strong> {results['config']['engine_version']} | <strong>Training Duration:</strong> {results['config']['training_duration']:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced hot topics detailed analysis
            st.markdown("### 🔥 Enhanced Hot Topics Analysis")
            
            for i, topic_info in enumerate(results['hot_topics']):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="hot-topic-card">
                            <h4>🔥 Hot Topic #{i+1}: Topic {topic_info['topic']}</h4>
                            <p><strong>🏷️ Keywords:</strong> {', '.join(topic_info['keywords'][:6])}</p>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px;">
                                <div><strong>🔥 Hotness:</strong> {topic_info['hotness_score']:.4f}</div>
                                <div><strong>📈 MAE:</strong> {topic_info['mae']:.4f}</div>
                                <div><strong>📊 RMSE:</strong> {topic_info['rmse']:.4f}</div>
                                <div><strong>🎯 MAPE:</strong> {topic_info['mape']:.1f}%</div>
                                <div><strong>📊 Avg Prob:</strong> {topic_info['avg_prob']:.4f}</div>
                                <div><strong>⚡ Performance:</strong> {'🏆 Excellent' if topic_info['mae'] < 0.01 else '👍 Good' if topic_info['mae'] < 0.02 else '📈 Fair'}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Enhanced mini performance chart for each topic
                        if st.session_state.predictions is not None and st.session_state.actuals is not None:
                            fig_mini = go.Figure()
                            
                            time_steps = np.arange(len(st.session_state.predictions))
                            
                            fig_mini.add_trace(go.Scatter(
                                x=time_steps,
                                y=st.session_state.actuals[:, i],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=4)
                            ))
                            
                            fig_mini.add_trace(go.Scatter(
                                x=time_steps,
                                y=st.session_state.predictions[:, i],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                marker=dict(size=4, symbol='square')
                            ))
                            
                            fig_mini.update_layout(
                                height=200,
                                margin=dict(l=10, r=10, t=30, b=10),
                                title=f"Topic {topic_info['topic']} Performance",
                                showlegend=False,
                                xaxis_title="Time",
                                yaxis_title="Probability"
                            )
                            
                            st.plotly_chart(fig_mini, use_container_width=True)
            
            # Enhanced comprehensive interactive visualizations
            st.markdown("### 📊 Enhanced Interactive AI Forecasting Dashboard")
            
            # Create comprehensive visualization with multiple subplots
            fig = make_subplots(
                rows=4, cols=2,
                subplot_titles=(
                    'Overall Hot Topics Performance (Enhanced)', 
                    'Prediction vs Actual Correlation Analysis',
                    'Individual Topic Performance Comparison', 
                    'Model Performance Ranking (MAE)',
                    'Enhanced Hotness Score Distribution', 
                    'Prediction Accuracy Evolution',
                    'Ensemble Model Contribution Analysis',
                    'Error Distribution & Statistical Analysis'
                ),
                specs=[
                    [{"colspan": 2}, None],
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "histogram"}]
                ]
            )
            
            if st.session_state.predictions is not None and st.session_state.actuals is not None:
                # 1. Overall performance trend (top row, full width)
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
                        marker=dict(size=8)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=pred_mean,
                        mode='lines+markers',
                        name='Predicted (Enhanced Ensemble)',
                        line=dict(color='#ff7f0e', width=3, dash='dash'),
                        marker=dict(size=8, symbol='square')
                    ),
                    row=1, col=1
                )
                
                # Add confidence intervals
                pred_std = st.session_state.predictions.std(axis=1)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([time_steps, time_steps[::-1]]),
                        y=np.concatenate([pred_mean + pred_std, (pred_mean - pred_std)[::-1]]),
                        fill='toself',
                        fillcolor='rgba(255,127,14,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Prediction Confidence',
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # 2. Prediction vs Actual correlation (row 2, left)
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state.actuals.flatten(),
                        y=st.session_state.predictions.flatten(),
                        mode='markers',
                        name='Predictions vs Actuals',
                        marker=dict(
                            color='rgba(50, 171, 96, 0.6)',
                            size=8,
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
                
                # 3. Topic performance ranking (row 2, right)
                topic_names = [f"Topic {t['topic']}" for t in results['hot_topics']]
                mae_values = [t['mae'] for t in results['hot_topics']]
                colors = ['#ff4444', '#44ff44', '#4444ff'][:len(mae_values)]
                
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
                
                # 4. Enhanced hotness distribution (row 3, left)
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
                
                # 5. Accuracy evolution (row 3, right)
                mae_by_time = [np.mean(np.abs(st.session_state.predictions[i] - st.session_state.actuals[i])) 
                              for i in range(len(st.session_state.predictions))]
                
                fig.add_trace(
                    go.Scatter(
                        x=time_steps,
                        y=mae_by_time,
                        mode='lines+markers',
                        name='MAE Evolution',
                        line=dict(color='#d62728', width=3),
                        marker=dict(size=6)
                    ),
                    row=3, col=2
                )
                
                # 6. Ensemble contribution analysis (row 4, left)
                ensemble_weights = results['config']['ensemble_weights']
                methods = list(ensemble_weights.keys())
                weights = list(ensemble_weights.values())
                
                fig.add_trace(
                    go.Bar(
                        x=methods,
                        y=weights,
                        name='Ensemble Weights',
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                        text=[f"{w:.2f}" for w in weights],
                        textposition='outside'
                    ),
                    row=4, col=1
                )
                
                # 7. Error distribution (row 4, right)
                errors = (st.session_state.predictions - st.session_state.actuals).flatten()
                
                fig.add_trace(
                    go.Histogram(
                        x=errors,
                        name='Error Distribution',
                        nbinsx=30,
                        marker_color='rgba(255, 100, 100, 0.7)',
                        opacity=0.8
                    ),
                    row=4, col=2
                )
            
            # Enhanced layout configuration
            fig.update_layout(
                height=1400,
                title_text=f"🔥 Enhanced GDELT Hot Topics AI Forecasting Results - User: {CURRENT_USER}",
                showlegend=True,
                title_font_size=18,
                title_x=0.5
            )
            
            # Update axis labels with enhanced styling
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
            
            fig.update_xaxes(title_text="Model Type", row=4, col=1)
            fig.update_yaxes(title_text="Weight", row=4, col=1)
            
            fig.update_xaxes(title_text="Prediction Error", row=4, col=2)
            fig.update_yaxes(title_text="Frequency", row=4, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced individual topic analysis
            st.markdown("### 🎯 Enhanced Individual Topic Deep Dive")
            
            selected_topic_idx = st.selectbox(
                "Select a hot topic for comprehensive analysis:",
                options=range(len(results['hot_topics'])),
                format_func=lambda x: f"Topic {results['hot_topics'][x]['topic']}: {', '.join(results['hot_topics'][x]['keywords'][:3])} (MAE: {results['hot_topics'][x]['mae']:.4f})"
            )
            
            if selected_topic_idx is not None:
                topic_info = results['hot_topics'][selected_topic_idx]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="hot-topic-card">
                        <h4>📊 Enhanced Topic {topic_info['topic']} Analysis</h4>
                        <p><strong>🏷️ Keywords:</strong> {', '.join(topic_info['keywords'])}</p>
                        <h5>📈 Performance Metrics:</h5>
                        <ul>
                            <li><strong>MAE:</strong> {topic_info['mae']:.4f} {'🏆' if topic_info['mae'] < 0.01 else '👍' if topic_info['mae'] < 0.02 else '📈'}</li>
                            <li><strong>RMSE:</strong> {topic_info['rmse']:.4f}</li>
                            <li><strong>MAPE:</strong> {topic_info['mape']:.1f}%</li>
                            <li><strong>Hotness Score:</strong> {topic_info['hotness_score']:.4f}</li>
                            <li><strong>Average Probability:</strong> {topic_info['avg_prob']:.4f}</li>
                        </ul>
                        <h5>🎯 Quality Assessment:</h5>
                        <p>{'🏆 Excellent forecasting performance!' if topic_info['mae'] < 0.01 else 
                           '👍 Good forecasting quality!' if topic_info['mae'] < 0.02 else 
                           '📈 Moderate performance - room for improvement.'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Enhanced individual topic visualization
                    if st.session_state.predictions is not None and st.session_state.actuals is not None:
                        fig_individual = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=[f'Topic {topic_info["topic"]} Forecast vs Actual', 'Prediction Error Analysis'],
                            vertical_spacing=0.15
                        )
                        
                        time_steps = np.arange(len(st.session_state.predictions))
                        
                        # Main prediction plot
                        fig_individual.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=st.session_state.actuals[:, selected_topic_idx],
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue', width=3),
                                marker=dict(size=6)
                            ),
                            row=1, col=1
                        )
                        
                        fig_individual.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=st.session_state.predictions[:, selected_topic_idx],
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red', width=3, dash='dash'),
                                marker=dict(size=6, symbol='square')
                            ),
                            row=1, col=1
                        )
                        
                        # Error analysis
                        errors = st.session_state.predictions[:, selected_topic_idx] - st.session_state.actuals[:, selected_topic_idx]
                        
                        fig_individual.add_trace(
                            go.Scatter(
                                x=time_steps,
                                y=errors,
                                mode='lines+markers',
                                name='Prediction Error',
                                line=dict(color='green', width=2),
                                marker=dict(size=4)
                            ),
                            row=2, col=1
                        )
                        
                        # Add zero line for error plot
                        fig_individual.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                        
                        fig_individual.update_layout(
                            height=500,
                            title=f"Enhanced Analysis: Topic {topic_info['topic']}",
                            showlegend=True
                        )
                        
                        fig_individual.update_xaxes(title_text="Time Steps", row=2, col=1)
                        fig_individual.update_yaxes(title_text="Topic Probability", row=1, col=1)
                        fig_individual.update_yaxes(title_text="Error", row=2, col=1)
                        
                        st.plotly_chart(fig_individual, use_container_width=True)
            
            # Enhanced model insights and recommendations
            st.markdown("### 💡 Enhanced AI Model Insights & Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Key Enhanced Findings")
                
                best_topic = min(results['hot_topics'], key=lambda x: x['mae'])
                worst_topic = max(results['hot_topics'], key=lambda x: x['mae'])
                
                st.markdown(f"""
                <div class="success-card">
                    <h5>🏆 Best Performing Topic</h5>
                    <p><strong>Topic {best_topic['topic']}:</strong> {', '.join(best_topic['keywords'][:3])}</p>
                    <p><strong>MAE:</strong> {best_topic['mae']:.4f} | <strong>MAPE:</strong> {best_topic['mape']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="warning-card">
                    <h5>📈 Most Challenging Topic</h5>
                    <p><strong>Topic {worst_topic['topic']}:</strong> {', '.join(worst_topic['keywords'][:3])}</p>
                    <p><strong>MAE:</strong> {worst_topic['mae']:.4f} | <strong>MAPE:</strong> {worst_topic['mape']:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced performance assessment
                avg_mae = results['overall']['mae']
                if avg_mae < 0.005:
                    st.success("🎉 **Outstanding AI Performance** (MAE < 0.005) - Production Ready!")
                elif avg_mae < 0.01:
                    st.success("🎉 **Excellent AI Performance** (MAE < 0.01) - Highly Accurate!")
                elif avg_mae < 0.02:
                    st.info("👍 **Good AI Performance** (MAE < 0.02) - Reliable Forecasting")
                elif avg_mae < 0.05:
                    st.warning("📈 **Moderate Performance** - Consider parameter tuning")
                else:
                    st.error("📉 **Performance needs improvement** - Review data quality and model configuration")
            
            with col2:
                st.markdown("#### 🔧 Enhanced Model Configuration Used")
                config = results['config']
                
                st.markdown(f"""
                <div class="file-info-card">
                    <h5>🎛️ AI Model Parameters</h5>
                    <ul>
                        <li><strong>Total Topics Discovered:</strong> {config['n_topics']}</li>
                        <li><strong>Hot Topics Analyzed:</strong> {config['top_k']}</li>
                        <li><strong>Forecast Horizon:</strong> {config['forecast_horizon']} days</li>
                        <li><strong>Training Duration:</strong> {config['training_duration']:.1f} seconds</li>
                        <li><strong>Engine Version:</strong> {config['engine_version']}</li>
                    </ul>
                    
                    <h5>⚖️ Enhanced Ensemble Weights</h5>
                    <ul>
                        <li><strong>Prophet (Time Series):</strong> {config['ensemble_weights']['prophet']:.2f}</li>
                        <li><strong>XGBoost (ML):</strong> {config['ensemble_weights']['xgboost']:.2f}</li>
                        <li><strong>LSTM (Deep Learning):</strong> {config['ensemble_weights']['lstm']:.2f}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced feature importance (if available)
                if hasattr(st.session_state.forecaster, 'feature_importance'):
                    st.markdown("**📊 Top XGBoost Features (Enhanced):**")
                    
                    # Aggregate feature importance across all topics
                    all_importance = {}
                    for topic_key, features in st.session_state.forecaster.feature_importance.items():
                        for feature, importance in features.items():
                            if feature not in all_importance:
                                all_importance[feature] = []
                            all_importance[feature].append(importance)
                    
                    avg_importance = {k: np.mean(v) for k, v in all_importance.items()}
                    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:8]
                    
                    for i, (feature, importance) in enumerate(top_features, 1):
                        st.write(f"   {i}. **{feature}:** {importance:.4f}")
            
            # Enhanced download section
            st.markdown("### 💾 Enhanced Download Results & Comprehensive Reports")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Enhanced predictions download
                if st.session_state.predictions is not None:
                    pred_df = pd.DataFrame(
                        st.session_state.predictions,
                        columns=[f"Topic_{results['hot_topics'][i]['topic']}_Prediction" 
                                for i in range(len(results['hot_topics']))]
                    )
                    
                    if 'test_dates' in results and len(results['test_dates']) == len(pred_df):
                        pred_df['Date'] = results['test_dates']
                        pred_df['User'] = CURRENT_USER
                        pred_df['Generated_Time'] = CURRENT_TIME
                        pred_df = pred_df[['Date', 'User', 'Generated_Time'] + [col for col in pred_df.columns if col not in ['Date', 'User', 'Generated_Time']]]
                    
                    csv_pred = pred_df.to_csv(index=False)
                    st.download_button(
                        "📈 Enhanced Predictions",
                        csv_pred,
                        f"gdelt_hot_topics_predictions_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        help="Download AI ensemble predictions for all hot topics"
                    )
            
            with col2:
                # Enhanced actual values download
                if st.session_state.actuals is not None:
                    actual_df = pd.DataFrame(
                        st.session_state.actuals,
                        columns=[f"Topic_{results['hot_topics'][i]['topic']}_Actual" 
                                for i in range(len(results['hot_topics']))]
                    )
                    
                    if 'test_dates' in results and len(results['test_dates']) == len(actual_df):
                        actual_df['Date'] = results['test_dates']
                        actual_df['User'] = CURRENT_USER
                        actual_df['Generated_Time'] = CURRENT_TIME
                        actual_df = actual_df[['Date', 'User', 'Generated_Time'] + [col for col in actual_df.columns if col not in ['Date', 'User', 'Generated_Time']]]
                    
                    csv_actual = actual_df.to_csv(index=False)
                    st.download_button(
                        "📊 Enhanced Actuals",
                        csv_actual,
                        f"gdelt_hot_topics_actuals_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        help="Download actual values for comprehensive comparison"
                    )
            
            with col3:
                # Enhanced performance report
                report_data = {
                    'Topic_ID': [t['topic'] for t in results['hot_topics']],
                    'Keywords': [', '.join(t['keywords']) for t in results['hot_topics']],
                    'MAE': [t['mae'] for t in results['hot_topics']],
                    'RMSE': [t['rmse'] for t in results['hot_topics']],
                    'MAPE': [t['mape'] for t in results['hot_topics']],
                    'Hotness_Score': [t['hotness_score'] for t in results['hot_topics']],
                    'Avg_Probability': [t['avg_prob'] for t in results['hot_topics']],
                    'Performance_Grade': ['Excellent' if t['mae'] < 0.01 else 'Good' if t['mae'] < 0.02 else 'Fair' for t in results['hot_topics']]
                }
                
                # Enhanced overall metrics
                overall_data = {
                    'Metric': ['Overall_MAE', 'Overall_RMSE', 'Overall_MAPE', 'Training_Duration_Seconds', 'Engine_Version'],
                    'Value': [results['overall']['mae'], results['overall']['rmse'], results['overall']['mape'], 
                             results['config']['training_duration'], results['config']['engine_version']]
                }
                
                report_df = pd.DataFrame(report_data)
                overall_df = pd.DataFrame(overall_data)
                
                # Enhanced comprehensive report
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                report_text = f"""ENHANCED GDELT Hot Topics AI Forecasting Report
=================================================
User: {CURRENT_USER}
Generated: {current_time} UTC
Report Type: Enhanced AI Analysis with Ensemble Forecasting

=== EXECUTIVE SUMMARY ===
Total Topics Analyzed: {config['n_topics']}
Hot Topics Focus: {config['top_k']}
Forecast Horizon: {config['forecast_horizon']} days
Training Duration: {config['training_duration']:.1f} seconds
Engine Version: {config['engine_version']}

=== OVERALL PERFORMANCE METRICS ===
{overall_df.to_string(index=False)}

=== HOT TOPICS DETAILED PERFORMANCE ===
{report_df.to_string(index=False)}

=== ENSEMBLE MODEL CONFIGURATION ===
Prophet Weight: {config['ensemble_weights']['prophet']:.2f}
XGBoost Weight: {config['ensemble_weights']['xgboost']:.2f}
LSTM Weight: {config['ensemble_weights']['lstm']:.2f}

=== PERFORMANCE ASSESSMENT ===
Best Topic: Topic {min(results['hot_topics'], key=lambda x: x['mae'])['topic']} (MAE: {min(results['hot_topics'], key=lambda x: x['mae'])['mae']:.4f})
Most Challenging: Topic {max(results['hot_topics'], key=lambda x: x['mae'])['topic']} (MAE: {max(results['hot_topics'], key=lambda x: x['mae'])['mae']:.4f})
Overall Grade: {'Excellent' if results['overall']['mae'] < 0.01 else 'Good' if results['overall']['mae'] < 0.02 else 'Fair'}

=== TECHNICAL METADATA ===
Training Records: {results['metadata']['train_records']:,}
Test Records: {results['metadata']['test_records']:,}
System User: {results['metadata']['user']}
Processing Timestamp: {results['metadata']['timestamp']}

=== RECOMMENDATIONS ===
{('• Model performance is excellent - ready for production deployment' if results['overall']['mae'] < 0.01 else 
  '• Model shows good performance - suitable for most forecasting tasks' if results['overall']['mae'] < 0.02 else 
  '• Consider parameter tuning or additional data for improved performance')}
• Monitor model performance regularly with new data
• Consider ensemble weight optimization for specific use cases
• Evaluate topic relevance periodically for domain changes

Report generated by Enhanced GDELT Hot Topics AI Forecaster
Powered by Prophet + XGBoost + LSTM Ensemble Architecture
"""
                
                st.download_button(
                    "📋 Enhanced Report",
                    report_text,
                    f"gdelt_enhanced_report_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    help="Download comprehensive AI performance report"
                )
            
            with col4:
                # Enhanced model configuration download
                config_data = {
                    'user': CURRENT_USER,
                    'timestamp': CURRENT_TIME,
                    'engine_version': 'Enhanced v2.0',
                    'model_config': results['config'],
                    'hot_topics': results['hot_topic_indices'],
                    'performance': {
                        'overall_mae': results['overall']['mae'],
                        'overall_rmse': results['overall']['rmse'],
                        'overall_mape': results['overall']['mape'],
                        'best_topic_mae': min([t['mae'] for t in results['hot_topics']]),
                        'worst_topic_mae': max([t['mae'] for t in results['hot_topics']]),
                        'performance_grade': 'Excellent' if results['overall']['mae'] < 0.01 else 'Good' if results['overall']['mae'] < 0.02 else 'Fair'
                    },
                    'topic_details': [{
                        'topic_id': t['topic'],
                        'keywords': t['keywords'],
                        'mae': t['mae'],
                        'rmse': t['rmse'],
                        'mape': t['mape'],
                        'hotness_score': t['hotness_score']
                    } for t in results['hot_topics']],
                    'system_info': {
                        'train_records': results['metadata']['train_records'],
                        'test_records': results['metadata']['test_records'],
                        'processing_timestamp': results['metadata']['timestamp']
                    }
                }
                
                import json
                config_json = json.dumps(config_data, indent=2, default=str)
                
                st.download_button(
                    "⚙️ Enhanced Config",
                    config_json,
                    f"gdelt_enhanced_config_{CURRENT_USER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    help="Download enhanced model configuration and metadata"
                )
            
            # Enhanced action buttons
            st.markdown("---")
            st.markdown("### 🚀 Enhanced Next Actions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 New Enhanced Analysis", type="secondary"):
                    # Reset session state
                    for key in list(st.session_state.keys()):
                        if key not in ['current_user', 'start_time']:
                            del st.session_state[key]
                    st.session_state.step = 1
                    st.success("✅ Ready for new enhanced analysis!")
                    st.rerun()
            
            with col2:
                if st.button("🔧 Retune Enhanced Model", type="secondary"):
                    # Go back to model training with current data
                    st.session_state.step = 4
                    st.session_state.model_trained = False
                    st.info("🔧 Ready for enhanced model retuning!")
                    st.rerun()
            
            with col3:
                if st.button("📊 Export Enhanced Dashboard", type="secondary"):
                    st.info("💡 Enhanced dashboard export feature - coming soon in v3.0!")
                    st.balloons()
            
            with col4:
                if st.button("🎭 Try Enhanced Demo", type="secondary"):
                    processor = OptimizedGDELTDataProcessor()
                    train_data, test_data = processor.create_demo_data()
                    
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    st.session_state.step = 4
                    st.success("✅ Enhanced demo data loaded!")
                    st.rerun()
    
    # Enhanced sidebar status
    with st.sidebar:
        st.markdown("## 📊 Enhanced Session Status")
        
        st.markdown(f"""
        <div class="user-info" style="margin-bottom: 1rem;">
            👤 <strong>{CURRENT_USER}</strong><br>
            🕐 Started: {CURRENT_TIME}<br>
            🔥 Engine: Enhanced v2.0
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced progress indicators
        if st.session_state.step >= 2 and st.session_state.zip_structure:
            st.success("✅ ZIP File Analyzed")
        
        if st.session_state.step >= 3 and st.session_state.train_data is not None:
            st.success("✅ Data Processed (Enhanced)")
            st.info(f"🏋️ Train: {len(st.session_state.train_data):,}")
            st.info(f"🧪 Test: {len(st.session_state.test_data):,}")
        
        if st.session_state.step >= 4 and st.session_state.model_trained:
            st.success("✅ AI Model Trained")
            if st.session_state.results:
                config = st.session_state.results['config']
                st.info(f"🔥 Hot Topics: {config['top_k']}")
                st.info(f"📅 Forecast: {config['forecast_horizon']} days")
                st.info(f"⏱️ Duration: {config['training_duration']:.1f}s")
        
        if st.session_state.step == 5 and st.session_state.results:
            st.success("✅ Enhanced Results Ready")
            mae = st.session_state.results['overall']['mae']
            st.metric("Overall MAE", f"{mae:.4f}")
            
            performance_grade = 'Excellent' if mae < 0.01 else 'Good' if mae < 0.02 else 'Fair'
            if performance_grade == 'Excellent':
                st.success(f"🏆 {performance_grade}")
            elif performance_grade == 'Good':
                st.info(f"👍 {performance_grade}")
            else:
                st.warning(f"📈 {performance_grade}")
        
        st.markdown("---")
        
        # Enhanced tips and help
        st.markdown("### 💡 Enhanced Quick Tips")
        
        if st.session_state.step == 1:
            st.info("📁 Upload ZIP files with enhanced processing engine")
        elif st.session_state.step == 2:
            st.info("🔍 Smart file categorization with AI assistance")
        elif st.session_state.step == 3:
            st.info("📊 Enhanced data processing with quality validation")
        elif st.session_state.step == 4:
            st.info("🔥 AI ensemble training for optimal performance")
        elif st.session_state.step == 5:
            st.info("📈 Comprehensive AI analysis with actionable insights")
        
        st.markdown("### 🚀 Enhanced Performance Tips")
        st.info("⚡ Enhanced demo data for instant testing")
        st.info("🎯 AI-optimized hot topic selection")
        st.info("💾 Smart memory management for large datasets")
        st.info("🔧 Advanced ensemble model configuration")
        
        # Enhanced system information
        if st.checkbox("🖥️ Show Enhanced System Info"):
            memory = psutil.virtual_memory()
            st.write(f"💾 **Memory:** {memory.percent:.1f}% used")
            st.write(f"💻 **Available:** {memory.available / (1024**3):.1f} GB")
            st.write(f"🖥️ **CPU Cores:** {psutil.cpu_count()}")
            st.write(f"🤖 **TensorFlow:** {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
            st.write(f"🐍 **Python:** Enhanced AI Stack")
            st.write(f"👤 **User:** {CURRENT_USER}")
            st.write(f"🕐 **Session:** {CURRENT_TIME}")
        
        # Emergency recovery tools
        emergency_recovery()

# Enhanced footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(90deg, #f8f9fa, #e9ecef); border-radius: 10px; margin: 2rem 0;">
    <h4 style="color: #FF4B4B; margin-bottom: 1rem;">🔥 Enhanced GDELT Hot Topics AI Forecaster</h4>
    <p><strong>Complete Enhanced Pipeline:</strong> Smart ZIP Upload → AI Data Processing → Advanced Topic Modeling → Ensemble AI Forecasting</p>
    <p><strong>User:</strong> {CURRENT_USER} | <strong>Session:</strong> {CURRENT_TIME} UTC | <strong>Engine:</strong> Enhanced v2.0</p>
    <p><strong>AI Architecture:</strong> Prophet + XGBoost + LSTM Ensemble | <strong>Powered by:</strong> Streamlit ⚡ + Advanced AI Stack 🤖</p>
    <p style="font-size: 0.9em; color: #999; margin-top: 1rem;">
        🎯 Enhanced hot topic detection | ⚡ Lightning-fast processing | 🚀 Production-ready AI | 🔧 Advanced configuration
    </p>
    <p style="font-size: 0.8em; color: #aaa; margin-top: 0.5rem;">
        Enhanced Version 2.0 - Built for {CURRENT_USER} | Generated: {CURRENT_TIME}
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()