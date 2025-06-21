"""
GDELT Hot Topics Forecaster - Optimized Version
User: strawberrymilktea0604
Current Time: 2025-06-21 10:50:55 UTC
Location: TungNguyen/demo/app.py
"""

# Import từ file optimized
import streamlit as st
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Page config
st.set_page_config(
    page_title="🔥 GDELT Hot Topics Forecaster - Optimized",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import optimized functions
from optimized_file_upload import (
    process_large_zip_chunked,
    smart_data_sampling,
    get_memory_usage,
    MemoryManager
)

def main():
    """Main optimized application"""
    
    st.title("🔥 GDELT Hot Topics Forecaster - Optimized")
    
    # User info
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #4CAF50, #45A049); color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 2rem;">
        👤 <strong>User:</strong> strawberrymilktea0604 | 
        🕐 <strong>Current Time:</strong> 2025-06-21 10:50:55 UTC | 
        📍 <strong>Location:</strong> TungNguyen/demo/app.py | 
        💾 <strong>Mode:</strong> Memory Optimized
    </div>
    """, unsafe_allow_html=True)
    
    # Memory monitoring
    with st.sidebar:
        st.markdown("## 📊 System Status")
        memory_usage = get_memory_usage()
        st.metric("💾 Memory Usage", f"{memory_usage:.1f} MB")
        
        if memory_usage > 500:
            st.warning("⚠️ High memory usage")
        else:
            st.success("✅ Memory usage OK")
    
    # Optimized file upload section
    st.markdown("### 📁 Optimized Large File Upload")
    
    uploaded_file = st.file_uploader(
        "Upload GDELT ZIP file (up to 2GB)",
        type=['zip'],
        help="Large files will be processed in chunks to prevent crashes"
    )
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 File Size", f"{file_size_mb:.1f} MB")
        with col2:
            processing_mode = "Chunked" if file_size_mb > 200 else "Standard"
            st.metric("🚀 Mode", processing_mode)
        with col3:
            est_time = max(1, int(file_size_mb / 20))
            st.metric("⏱️ Est. Time", f"{est_time} min")
        
        if st.button("🚀 Process File (Optimized)", type="primary"):
            with st.spinner("Processing large file... Please wait..."):
                
                # Process with optimized function
                processed_df = process_large_zip_chunked(uploaded_file)
                
                if processed_df is not None:
                    # Smart sampling if too large
                    if len(processed_df) > 50000:
                        processed_df = smart_data_sampling(processed_df, 50000)
                    
                    st.session_state['processed_data'] = processed_df
                    st.success(f"✅ Successfully processed {len(processed_df):,} rows!")
                    
                    # Show preview
                    st.dataframe(processed_df.head(), use_container_width=True)
                    
                    # Continue with your existing analysis code here...
                    st.info("🔄 Ready for topic analysis and forecasting!")

if __name__ == "__main__":
    main()