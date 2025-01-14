import streamlit as st
import math
import logging
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from huggingface_hub import login
from smolagents import tool, CodeAgent, HfApiModel, ManagedAgent, ToolCallingAgent
from typing import Dict, List  # Added this import


st.set_page_config(page_title="Advanced Geotechnical AI", layout="wide")

# Use Streamlit secrets
try:
    #login(st.secrets["HUGGINGFACE_API_KEY"])
    hf_key = st.secrets["HUGGINGFACE_API_KEY"]
except Exception as e:
    st.error("Authentication failed. Please check API key configuration.")
    st.stop()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Styling
st.markdown("""
<style>
.main {background-color: #f5f5f5}
.stButton>button {
    background-color: #0066cc;
    color: white;
    border-radius: 5px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar tools
with st.sidebar:
    st.title("🔧 Analysis Tools")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Soil Classification", "Tunnel Support", "RMR Analysis", 
         "TBM Performance", "Face Stability"]
    )
    
    with st.expander("Analysis Parameters"):
        if analysis_type == "Soil Classification":
            soil_type = st.selectbox("Soil Type", ["clay", "sand", "silt"])
            plasticity_index = st.number_input("Plasticity Index", 0.0, 100.0, 25.0)
            liquid_limit = st.number_input("Liquid Limit", 0.0, 100.0, 50.0)
            
        elif analysis_type == "Tunnel Support":
            depth = st.number_input("Depth (m)", 0.0, 1000.0, 100.0)
            soil_density = st.number_input("Soil Density (kg/m³)", 1000.0, 3000.0, 1800.0)
            k0 = st.number_input("K₀ Coefficient", 0.0, 2.0, 0.5)
            tunnel_diameter = st.number_input("Tunnel Diameter (m)", 1.0, 20.0, 6.0)

# Main content
st.title("🏗️ Geotechnical AI Assistant")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    user_input = st.text_input("Ask a question:")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Processing..."):
            response = process_request(managed_geotech_agent, user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {msg['content']}")

with col2:
    st.subheader("📊 Analysis Results")
    if st.session_state.current_analysis:
        with st.expander("Detailed Results", expanded=True):
            st.json(st.session_state.current_analysis)
        
        if analysis_type == "Tunnel Support":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0, tunnel_diameter],
                y=[depth, depth],
                mode='lines',
                name='Tunnel Level'
            ))
            fig.update_layout(
                title="Tunnel Cross Section",
                xaxis_title="Width (m)",
                yaxis_title="Depth (m)",
                yaxis_autorange="reversed"
            )
            st.plotly_chart(fig)

@st.cache_resource
def initialize_agents():
    model = HfApiModel("mistralai/Mistral-Nemo-Instruct-2407")
    
    @tool
    def classify_soil(soil_type: str, plasticity_index: float, liquid_limit: float) -> Dict:
        if soil_type.lower() == 'clay':
            if plasticity_index > 50:
                return {"classification": "CH", "description": "High plasticity clay"}
            elif plasticity_index > 30:
                return {"classification": "CI", "description": "Medium plasticity clay"}
            else:
                return {"classification": "CL", "description": "Low plasticity clay"}
        return {"classification": "Unknown", "description": "Unknown soil type"}
    
    @tool
    def calculate_tunnel_support(depth: float, soil_density: float, k0: float, tunnel_diameter: float) -> Dict:
        g = 9.81
        vertical_stress = depth * soil_density * g / 1000
        horizontal_stress = k0 * vertical_stress
        support_pressure = (vertical_stress + horizontal_stress) / 2
        safety_factor = 1.5 if depth < 30 else 2.0
        
        return {
            "support_pressure": round(support_pressure, 2),
            "design_pressure": round(support_pressure * safety_factor, 2),
            "safety_factor": safety_factor,
            "vertical_stress": round(vertical_stress, 2),
            "horizontal_stress": round(horizontal_stress, 2)
        }
    
    geotech_agent = ToolCallingAgent(
        tools=[classify_soil, calculate_tunnel_support],
        model=model,
        max_steps=10
    )
    
    return ManagedAgent(
        agent=geotech_agent,
        name="geotech_analysis",
        description="Performs geotechnical calculations and analysis."
    )

managed_geotech_agent = initialize_agents()

def process_request(agent: ManagedAgent, request: str):
    try:
        result = agent(request=request)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Error: {str(e)}"

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")
