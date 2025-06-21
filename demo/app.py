"""
GDELT Hot Topics Forecaster - Fixed Import Version
User: strawberrymilktea0604
Current Date and Time: 2025-06-21 12:04:30 UTC
Fixed: Import errors resolved
"""

import streamlit as st
import sys
from pathlib import Path
import gc

# Page config
st.set_page_config(
    page_title="🔥 GDELT Hot Topics Forecaster - Fixed",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import optimized functions - FIXED IMPORTS
try:
    from optimized_file_upload import (
        process_large_zip_chunked,
        smart_data_sampling,
        get_memory_usage,
        MemoryManager,
        optimized_file_upload_section
    )
    import_success = True
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    import_success = False

def show_error_fix():
    """Show error and fix information"""
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #FF5722, #E64A19); color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 2rem;">
        🚨 <strong>Import Error Fixed</strong><br>
        👤 <strong>User:</strong> strawberrymilktea0604 | 
        🕐 <strong>Time:</strong> 2025-06-21 12:04:30 UTC<br>
        ✅ <strong>Status:</strong> Function names now match
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application with fixed imports"""
    
    st.title("🔥 GDELT Hot Topics Forecaster - Fixed Version")
    
    show_error_fix()
    
    if import_success:
        st.success("✅ All imports successful!")
        
        # User info
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #4CAF50, #45A049); color: white; padding: 1rem; border-radius: 8px; text-align: center; margin-bottom: 2rem;">
            👤 <strong>User:</strong> strawberrymilktea0604 | 
            🕐 <strong>Current Time:</strong> 2025-06-21 12:04:30 UTC | 
            📍 <strong>Status:</strong> Fixed and Ready | 
            🚀 <strong>Mode:</strong> Memory Optimized
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
            🔧 **Fixed Features:**
            - ✅ All functions imported correctly
            - ✅ Memory management optimized
            - ✅ Chunked processing enabled
            - ✅ Smart data sampling active
            """)
        
        # Main upload section
        uploaded_file = optimized_file_upload_section()
        
        # Show next steps if data processed
        if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
            st.markdown("### 🚀 Next Steps")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎯 Topic Analysis"):
                    st.info("🔄 Topic analysis ready to implement...")
            
            with col2:
                if st.button("📈 Generate Forecasts"):
                    st.info("🔄 Forecasting ready to implement...")
            
            with col3:
                if st.button("📊 Visualizations"):
                    st.info("🔄 Visualizations ready to implement...")
    
    else:
        st.error("❌ Import failed - please check optimized_file_upload.py file")
        
        st.markdown("### 🔧 To Fix:")
        st.code("""
        1. Update optimized_file_upload.py with the fixed version above
        2. Commit and push changes to GitHub
        3. Streamlit will auto-reload
        """, language='text')
    
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