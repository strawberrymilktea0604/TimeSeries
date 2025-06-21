"""
Fixed Optimized File Upload for GDELT Hot Topics Forecaster
User: strawberrymilktea0604
Current Date and Time: 2025-06-21 12:04:30 UTC
Fixed: Import errors - function names now match
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import gc
import time
from pathlib import Path
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except:
        return 0.0

@st.cache_data(show_spinner=False)
def process_large_zip_chunked(uploaded_file, chunk_size_mb=50):
    """Process large ZIP file in chunks to avoid memory issues"""
    
    try:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        st.info(f"📁 Processing file: {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Create progress containers
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Read ZIP file
        status_text.text("🔍 Reading ZIP file...")
        zip_buffer = io.BytesIO(uploaded_file.getvalue())
        
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            file_list = zip_file.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            status_text.text(f"📋 Found {len(csv_files)} CSV files")
            
            processed_data = []
            total_files = len(csv_files)
            
            if total_files == 0:
                st.error("❌ No CSV files found in ZIP")
                return None
            
            # Process files in batches
            batch_size = min(5, total_files)  # Process max 5 files at a time
            
            for i in range(0, min(total_files, 20), batch_size):  # Limit to 20 files max
                batch_files = csv_files[i:i+batch_size]
                
                for j, csv_file in enumerate(batch_files):
                    current_file = i + j + 1
                    progress = current_file / min(total_files, 20)
                    
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Processing file {current_file}/{min(total_files, 20)}: {csv_file}")
                    
                    try:
                        # Read CSV in chunks
                        with zip_file.open(csv_file) as file:
                            # Read only first 5000 rows to avoid memory issues
                            df_chunk = pd.read_csv(file, nrows=5000, low_memory=False)
                            
                            # Keep only essential columns if they exist
                            essential_cols = ['DATE', 'Actor1Name', 'Actor2Name', 'EventCode', 'GoldsteinScale', 'SOURCEURL']
                            available_cols = [col for col in essential_cols if col in df_chunk.columns]
                            
                            if available_cols:
                                df_chunk = df_chunk[available_cols]
                                processed_data.append(df_chunk)
                            
                            # Memory cleanup
                            del df_chunk
                            gc.collect()
                            
                    except Exception as e:
                        st.warning(f"⚠️ Skipped {csv_file}: {str(e)}")
                        continue
                
                # Memory cleanup after each batch
                gc.collect()
                
                # Show memory usage
                memory_mb = get_memory_usage()
                with status_container:
                    st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")
            
            # Combine all processed data
            status_text.text("🔗 Combining processed data...")
            
            if processed_data:
                combined_df = pd.concat(processed_data, ignore_index=True)
                
                # Final cleanup
                del processed_data
                gc.collect()
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing completed!")
                
                return combined_df
            else:
                st.error("❌ No valid data found in ZIP file")
                return None
                
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return None

def smart_data_sampling(df, max_rows=50000):
    """Smart sampling to reduce data size while preserving patterns"""
    
    if df is None or len(df) <= max_rows:
        return df
    
    st.info(f"📉 Sampling data from {len(df):,} to {max_rows:,} rows for performance")
    
    try:
        # Stratified sampling by date if DATE column exists
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE'])
            
            if len(df) > 0:
                # Sample proportionally from each month
                df['YearMonth'] = df['DATE'].dt.to_period('M')
                sample_df = df.groupby('YearMonth').apply(
                    lambda x: x.sample(min(len(x), max_rows // max(12, df['YearMonth'].nunique())))
                ).reset_index(drop=True)
                
                return sample_df.head(max_rows)
        
        # Random sampling as fallback
        return df.sample(min(len(df), max_rows)).reset_index(drop=True)
        
    except Exception as e:
        st.warning(f"⚠️ Sampling error: {str(e)}, using random sample")
        return df.sample(min(len(df), max_rows)).reset_index(drop=True)

class MemoryManager:
    """Memory management for large file processing"""
    
    @staticmethod
    def get_memory_info():
        """Get current memory usage information"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Physical memory
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
                'percent': process.memory_percent()
            }
        except:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup"""
        gc.collect()
        
        # Clear pandas cache if possible
        try:
            if hasattr(pd, 'core'):
                if hasattr(pd.core, 'common'):
                    if hasattr(pd.core.common, 'clear_cache'):
                        pd.core.common.clear_cache()
        except:
            pass
    
    @staticmethod
    def check_memory_threshold(threshold_mb=800):
        """Check if memory usage exceeds threshold"""
        memory_info = MemoryManager.get_memory_info()
        return memory_info['rss_mb'] > threshold_mb
    
    @staticmethod
    def optimize_dataframe(df, max_memory_mb=100):
        """Optimize DataFrame memory usage"""
        
        if df is None:
            return None
            
        try:
            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() < len(df) * 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
            
            # Downcast numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Check if still too large
            current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            if current_memory > max_memory_mb:
                # Sample data if still too large
                sample_ratio = max_memory_mb / current_memory
                sample_size = int(len(df) * sample_ratio)
                df = df.sample(min(sample_size, len(df))).reset_index(drop=True)
                
                st.warning(f"⚠️ Data sampled to {len(df):,} rows to fit memory limit")
            
            return df
            
        except Exception as e:
            st.warning(f"⚠️ DataFrame optimization failed: {str(e)}")
            return df

# Memory monitoring decorator
def monitor_memory(func):
    """Decorator to monitor memory usage of functions"""
    def wrapper(*args, **kwargs):
        try:
            initial_memory = get_memory_usage()
            
            result = func(*args, **kwargs)
            
            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            if memory_diff > 50:  # More than 50MB increase
                st.warning(f"⚠️ Function used {memory_diff:.1f} MB memory")
            
            return result
        except Exception as e:
            st.error(f"❌ Memory monitoring error: {str(e)}")
            return func(*args, **kwargs)
    
    return wrapper

# Main optimized upload function
def optimized_file_upload_section():
    """Main optimized file upload section"""
    
    st.markdown("### 📁 Optimized Large File Upload")
    
    # Memory monitoring
    with st.sidebar:
        st.markdown("## 📊 System Status")
        memory_usage = get_memory_usage()
        st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
        
        if memory_usage > 500:
            st.warning("⚠️ High memory usage")
        else:
            st.success("✅ Memory usage OK")
    
    # Upload section
    st.warning("""
    ⚠️ **Large File Handling:**
    - Files will be processed in chunks
    - Data will be sampled to prevent crashes
    - Processing may take several minutes
    """)
    
    uploaded_file = st.file_uploader(
        "Upload GDELT ZIP file (optimized for large files)",
        type=['zip'],
        help="Large files will be automatically optimized during processing"
    )
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        # File size analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📁 File Size", f"{file_size_mb:.1f} MB")
        
        with col2:
            processing_mode = "Chunked" if file_size_mb > 200 else "Standard"
            st.metric("🚀 Processing Mode", processing_mode)
        
        with col3:
            estimated_time = max(1, int(file_size_mb / 20))  # ~20MB per minute
            st.metric("⏱️ Est. Time", f"{estimated_time} min")
        
        # Processing options
        max_rows = st.select_slider(
            "Maximum rows to process",
            options=[10000, 25000, 50000],
            value=25000,
            help="Reduce to improve performance"
        )
        
        # Process file button
        if st.button("🚀 Process File (Optimized)", type="primary"):
            
            with st.spinner("Processing large file... Please wait..."):
                
                # Clear memory before processing
                gc.collect()
                
                # Process file
                processed_df = process_large_zip_chunked(uploaded_file)
                
                if processed_df is not None:
                    # Smart sampling
                    if len(processed_df) > max_rows:
                        processed_df = smart_data_sampling(processed_df, max_rows)
                    
                    # Store in session state
                    st.session_state['processed_data'] = processed_df
                    
                    st.success(f"✅ Successfully processed {len(processed_df):,} rows!")
                    
                    # Show preview
                    st.markdown("### 👀 Data Preview")
                    st.dataframe(processed_df.head(), use_container_width=True)
                    
                    # Basic statistics
                    st.markdown("### 📊 Basic Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Rows", f"{len(processed_df):,}")
                    
                    with col2:
                        st.metric("Columns", len(processed_df.columns))
                    
                    with col3:
                        memory_size = processed_df.memory_usage(deep=True).sum() / 1024 / 1024
                        st.metric("Data Size", f"{memory_size:.1f} MB")
    
    return uploaded_file

if __name__ == "__main__":
    st.title("🔧 Optimized File Upload - Standalone Test")
    optimized_file_upload_section()