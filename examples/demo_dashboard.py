"""
SFM-2 Open Source Release
This file contains the public architecture and methodology.
For production deployment, additional private components are required.
"""

"""
Phase 6: Demo Dashboard for Model Comparison
Streamlit dashboard to compare Sona AI model outputs, fallback logs, and latency stats.
"""
import streamlit as st
import requests
import time
import json

st.set_page_config(page_title="SonaCore AI Pipeline", layout="wide")
st.title("🚀 SonaCore AI Pipeline Demo")
st.markdown("> **Live Demo**: Testing SFM-2 model training, API endpoints, and intelligent fallback system")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")
api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

# Health Check Section
st.header("📊 Model Health Status")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Health Status"):
        try:
            health_resp = requests.get(f"{api_base}/health", timeout=5)
            if health_resp.status_code == 200:
                health_data = health_resp.json()
                st.success("✅ API Connected")
                
                # Display model status
                for model_name, status in health_data.items():
                    if model_name in ['gpt2_lora', 'sfm2']:
                        icon = "🟢" if status.get('healthy', False) else "🔴"
                        st.write(f"{icon} **{model_name.upper()}**: {'Healthy' if status.get('healthy', False) else 'Unavailable'}")
                    elif model_name == 'openai':
                        icon = "🟢" if status.get('quota_ok', False) else "🟡"
                        availability = "Available" if status.get('available', False) else "Disabled"
                        quota = "OK" if status.get('quota_ok', False) else "No API Key"
                        st.write(f"{icon} **OpenAI Fallback**: {availability} ({quota})")
            else:
                st.error(f"❌ API Error: {health_resp.status_code}")
        except Exception as e:
            st.error(f"❌ Connection Failed: {e}")

with col2:
    st.info("💡 **Setup Instructions**")
    st.markdown("""
    - **SFM-2**: Run training script to load model
    - **GPT-2 LoRA**: Load pre-trained adapter
    - **OpenAI**: Set `OPENAI_API_KEY` environment variable
    """)

# Main Inference Section
st.header("🧠 Model Inference")
prompt = st.text_area("Enter your prompt:", height=100, placeholder="e.g., 'Create a Sona function to calculate fibonacci numbers'")
    
col1, col2, col3 = st.columns(3)
with col1:
    prompt_type = st.selectbox("Prompt Type", ["natural", "sona", "repl"], help="Type of prompt for intelligent routing")
with col2:
    complexity = st.selectbox("Complexity", ["auto", "simple", "complex"], help="Complexity level for model selection")
with col3:
    st.write("")  # Spacer
    st.write("")  # Spacer

if st.button("🚀 Run Inference", type="primary") and prompt.strip():
    st.info("🔄 Running inference...")
    t0 = time.time()
    
    try:
        resp = requests.post(
            f"{api_base}/inference",
            json={
                "prompt": prompt, 
                "prompt_type": prompt_type, 
                "complexity": complexity
            },
            timeout=30
        )
        latency = time.time() - t0
        
        if resp.status_code == 200:
            data = resp.json()
            
            # Check if it's a successful response or fallback error
            if data.get('success') == False:
                # Handle structured fallback response
                st.warning("⚠️ No models available - showing fallback response")
                st.error(f"**Error**: {data.get('message', 'Unknown error')}")
                st.write(f"**Error Code**: {data.get('error_code', 'N/A')}")
                st.write(f"**Fallback Used**: {data.get('fallback_used', 'none')}")
                if data.get('retry_suggested'):
                    st.info("💡 **Suggestion**: Configure an OpenAI API key or load a model to get actual responses")
            else:
                # Handle successful model response
                model_used = data.get('model', 'unknown')
                result = data.get('result', 'No result')
                
                # Display results with appropriate styling
                st.success(f"✅ **Model Used**: {model_used.upper()}")
                
                if model_used == 'openai':
                    st.info("🤖 Generated using OpenAI fallback")
                elif model_used == 'sfm2':
                    st.success("🎯 Generated using SFM-2 (Sona specialist)")
                elif model_used == 'gpt2_lora':
                    st.success("🔧 Generated using GPT-2 LoRA")
                
                # Display the generated code/text
                if prompt_type == 'sona':
                    st.code(result, language='python')
                else:
                    st.code(result)
                
                st.metric("⏱️ Latency", f"{latency:.2f}s")
                
        else:
            st.error(f"❌ API Error: HTTP {resp.status_code}")
            try:
                error_data = resp.json()
                st.write(error_data)
            except:
                st.write(resp.text)
                
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. Try again or check if the API server is running.")
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 Cannot connect to API at {api_base}. Is the server running?")
    except Exception as e:
        st.error(f"❌ Request failed: {e}")

# Instructions and Status
st.markdown("---")
st.header("📚 Quick Start Guide")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **🎯 Testing the Pipeline:**
    1. **Check Health**: Verify API connectivity
    2. **Test Prompts**: Try different prompt types
    3. **Enable Fallback**: Set OpenAI key for live responses
    
    **🔧 Example Prompts:**
    - `"Create a Sona function to sort an array"`
    - `"Write Python code for file reading"`
    - `"Explain how variables work in Sona"`
    """)

with col2:
    st.markdown("""
    **🚀 Production Setup:**
    ```bash
    # 1. Train SFM-2 model
    python scripts/train_sfm2.py
    
    # 2. Set OpenAI key (optional)
    set OPENAI_API_KEY=your_key_here
    
    # 3. Start API server
    python -m uvicorn api.app:app --reload
    ```
    """)

st.markdown("---")
st.markdown("### 📋 System Architecture")
st.markdown("""
**Intelligent Model Routing:**
- 🎯 **SFM-2**: Specialized for Sona language tasks
- 🔧 **GPT-2 LoRA**: Fine-tuned for general programming
- 🤖 **OpenAI GPT-3.5**: Fallback for maximum reliability

**Current Status**: ✅ API Infrastructure Ready | 🔧 Model Integration In Progress
""")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**SonaCore AI Pipeline v1.0**")
st.sidebar.markdown("Built for stakeholder demos and production deployment")
