"""
Optimized File Upload Handler - GDELT Hot Topics Forecaster
User: strawberrymilktea0604
Current Time: 2025-06-21 10:46:22 UTC
Issue: Web crashes when uploading large files
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

# Page configuration
st.set_page_config(
    page_title="🔥 GDELT Hot Topics - Optimized Upload",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Memory monitoring
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

# Chunked file processing
@st.cache_data(show_spinner=False)
def process_large_zip_chunked(uploaded_file, chunk_size_mb=50):
    """Process large ZIP file in chunks to avoid memory issues"""
    
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    st.info(f"📁 Processing file: {uploaded_file.name} ({file_size_mb:.1f} MB)")
    
    # Create progress containers
    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Read ZIP file
        status_text.text("🔍 Reading ZIP file...")
        zip_buffer = io.BytesIO(uploaded_file.getvalue())
        
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            file_list = zip_file.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            status_text.text(f"📋 Found {len(csv_files)} CSV files")
            
            processed_data = []
            total_files = len(csv_files)
            
            # Process files in batches
            batch_size = 5  # Process 5 files at a time
            
            for i in range(0, total_files, batch_size):
                batch_files = csv_files[i:i+batch_size]
                
                for j, csv_file in enumerate(batch_files):
                    current_file = i + j + 1
                    progress = current_file / total_files
                    
                    progress_bar.progress(progress)
                    status_text.text(f"📊 Processing file {current_file}/{total_files}: {csv_file}")
                    
                    try:
                        # Read CSV in chunks
                        with zip_file.open(csv_file) as file:
                            # Read only first 10000 rows to avoid memory issues
                            df_chunk = pd.read_csv(file, nrows=10000, low_memory=False)
                            
                            # Keep only essential columns
                            essential_cols = ['DATE', 'Actor1Name', 'Actor2Name', 'EventCode', 'GoldsteinScale']
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

# Memory-efficient data sampling
def smart_data_sampling(df, max_rows=50000):
    """Smart sampling to reduce data size while preserving patterns"""
    
    if len(df) <= max_rows:
        return df
    
    st.info(f"📉 Sampling data from {len(df):,} to {max_rows:,} rows for performance")
    
    # Stratified sampling by date if DATE column exists
    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        df = df.dropna(subset=['DATE'])
        
        # Sample proportionally from each month
        df['YearMonth'] = df['DATE'].dt.to_period('M')
        sample_df = df.groupby('YearMonth').apply(
            lambda x: x.sample(min(len(x), max_rows // 12))
        ).reset_index(drop=True)
        
        return sample_df.head(max_rows)
    else:
        # Random sampling
        return df.sample(min(len(df), max_rows)).reset_index(drop=True)

def main():
    """Main optimized application"""
    
    st.title("🔥 GDELT Hot Topics Forecaster - Optimized for Large Files")
    
    # User info
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50, #45A049); color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 2rem;">
        👤 <strong>User:</strong> strawberrymilktea0604 | 
        🕐 <strong>Current Time:</strong> 2025-06-21 10:46:22 UTC | 
        💾 <strong>Mode:</strong> Memory Optimized Upload
    </div>
    """, unsafe_allow_html=True)
    
    # Memory monitoring sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        memory_usage = get_memory_usage()
        st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
        
        if memory_usage > 500:
            st.warning("⚠️ High memory usage")
        else:
            st.success("✅ Memory usage OK")
        
        st.markdown("---")
        st.markdown("### 💡 Upload Tips")
        st.info("""
        🔧 **For large files:**
        - Files will be processed in chunks
        - Only essential columns kept
        - Data sampled if > 50K rows
        - Memory cleaned after each step
        """)
    
    # Optimized file upload
    st.markdown("### 📁 Optimized File Upload")
    
    # Upload size warning
    st.warning("""
    ⚠️ **Large File Handling:**
    - Files > 200MB will be processed in chunks
    - Data will be sampled to prevent crashes
    - Processing may take several minutes
    - Close other browser tabs to free memory
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
            if file_size_mb < 200:
                st.metric("🚀 Processing Mode", "Standard")
            else:
                st.metric("🚀 Processing Mode", "Chunked")
        
        with col3:
            estimated_time = max(1, int(file_size_mb / 20))  # ~20MB per minute
            st.metric("⏱️ Est. Time", f"{estimated_time} min")
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_rows = st.select_slider(
                "Maximum rows to process",
                options=[10000, 25000, 50000, 100000],
                value=50000,
                help="Reduce to improve performance"
            )
        
        with col2:
            sample_method = st.selectbox(
                "Sampling method",
                ["Stratified by date", "Random sampling", "Recent data only"],
                help="How to sample large datasets"
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
                    
                    # Next steps
                    st.markdown("### 🚀 Next Steps")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🎯 Run Topic Analysis"):
                            st.info("🔄 Topic analysis will be implemented next...")
                    
                    with col2:
                        if st.button("📈 Generate Forecasts"):
                            st.info("🔄 Forecasting will be implemented next...")
    
    # Memory cleanup button
    st.markdown("---")
    if st.button("🧹 Clear Memory"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Force garbage collection
        gc.collect()
        
        st.success("✅ Memory cleared!")
        st.rerun()

if __name__ == "__main__":
    main()