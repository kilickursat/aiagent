import streamlit as st
import math
import logging
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from huggingface_hub import login
from typing import Dict, List
from smolagents import tool, CodeAgent, HfApiModel, ManagedAgent, ToolCallingAgent

# Page configuration
st.set_page_config(
    page_title="Advanced Geotechnical AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
.stTextInput>div>div>input {border-radius: 5px}
div.stButton > button:hover {background-color: #0052a3}
.sidebar .sidebar-content {background-color: #f0f2f6}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}

@st.cache_resource
def initialize_agents():
    try:
        # Get API key from secrets
        hf_key = st.secrets["HUGGINGFACE_API_KEY"]
        login(hf_key)
        
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
            elif soil_type.lower() == 'sand':
                return {"classification": "SP", "description": "Poorly graded sand"}
            elif soil_type.lower() == 'silt':
                return {"classification": "ML", "description": "Low plasticity silt"}
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

        @tool
        def calculate_rmr(ucs: float, rqd: float, spacing: float, condition: int, 
                         groundwater: int, orientation: int) -> Dict:
            ratings = {
                "ucs": min(15, max(0, int(ucs/20))),
                "rqd": min(20, max(3, int(rqd/5))),
                "spacing": min(20, max(5, int(spacing*10))),
                "condition": min(30, max(0, condition)),
                "groundwater": min(15, max(0, groundwater)),
                "orientation": min(0, max(-12, orientation))
            }
            
            total_rmr = sum(ratings.values())
            
            if total_rmr > 80:
                rock_class = "I - Very good rock"
            elif total_rmr > 60:
                rock_class = "II - Good rock"
            elif total_rmr > 40:
                rock_class = "III - Fair rock"
            elif total_rmr > 20:
                rock_class = "IV - Poor rock"
            else:
                rock_class = "V - Very poor rock"
                
            return {
                "rmr_value": total_rmr,
                "rock_class": rock_class,
                "ratings": ratings
            }

        @tool
        def estimate_tbm_performance(ucs: float, rqd: float, joint_spacing: float,
                                   abrasivity: float, diameter: float) -> Dict:
            pr = 20 * (1/ucs) * (rqd/100) * (1/abrasivity)
            utilization = 0.85 - (0.01 * (abrasivity/2))
            advance_rate = pr * utilization * 24
            cutter_life = 100 * (250/ucs) * (2/abrasivity)

            return {
                "penetration_rate": round(pr, 2),
                "daily_advance": round(advance_rate, 2),
                "utilization": round(utilization * 100, 1),
                "cutter_life_hours": round(cutter_life, 0)
            }
        
        geotech_agent = ToolCallingAgent(
            tools=[classify_soil, calculate_tunnel_support, calculate_rmr, estimate_tbm_performance],
            model=model,
            max_steps=10
        )
        
        return ManagedAgent(
            agent=geotech_agent,
            name="geotech_analysis",
            description="Performs geotechnical calculations and analysis."
        )
    except Exception as e:
        st.error(f"Failed to initialize agents: {str(e)}")
        return None

def process_request(agent: ManagedAgent, request: str):
    try:
        result = agent(request=request)
        return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize agent
managed_geotech_agent = initialize_agents()

# Sidebar
with st.sidebar:
    st.title("🔧 Analysis Tools")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Soil Classification", "Tunnel Support", "RMR Analysis", "TBM Performance"]
    )
    
    with st.expander("Analysis Parameters", expanded=True):
        if analysis_type == "Soil Classification":
            st.session_state.analysis_params = {
                "soil_type": st.selectbox("Soil Type", ["clay", "sand", "silt"]),
                "plasticity_index": st.number_input("Plasticity Index", 0.0, 100.0, 25.0),
                "liquid_limit": st.number_input("Liquid Limit", 0.0, 100.0, 50.0)
            }
            
        elif analysis_type == "Tunnel Support":
            st.session_state.analysis_params = {
                "depth": st.number_input("Depth (m)", 0.0, 1000.0, 100.0),
                "soil_density": st.number_input("Soil Density (kg/m³)", 1000.0, 3000.0, 1800.0),
                "k0": st.number_input("K₀ Coefficient", 0.0, 2.0, 0.5),
                "tunnel_diameter": st.number_input("Tunnel Diameter (m)", 1.0, 20.0, 6.0)
            }
            
        elif analysis_type == "RMR Analysis":
            st.session_state.analysis_params = {
                "ucs": st.number_input("UCS (MPa)", 0.0, 250.0, 100.0),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0),
                "spacing": st.number_input("Joint Spacing (m)", 0.0, 2.0, 0.6),
                "condition": st.slider("Joint Condition", 0, 30, 15),
                "groundwater": st.slider("Groundwater Condition", 0, 15, 10),
                "orientation": st.slider("Joint Orientation", -12, 0, -5)
            }
            
        elif analysis_type == "TBM Performance":
            st.session_state.analysis_params = {
                "ucs": st.number_input("UCS (MPa)", 0.0, 250.0, 100.0),
                "rqd": st.number_input("RQD (%)", 0.0, 100.0, 75.0),
                "joint_spacing": st.number_input("Joint Spacing (m)", 0.0, 2.0, 0.6),
                "abrasivity": st.number_input("Cerchar Abrasivity Index", 0.0, 6.0, 2.0),
                "diameter": st.number_input("TBM Diameter (m)", 1.0, 15.0, 6.0)
            }
    
    if st.button("Run Analysis"):
        with st.spinner("Processing..."):
            if analysis_type == "Soil Classification":
                st.session_state.current_analysis = classify_soil(
                    st.session_state.analysis_params["soil_type"],
                    st.session_state.analysis_params["plasticity_index"],
                    st.session_state.analysis_params["liquid_limit"]
                )
            elif analysis_type == "Tunnel Support":
                st.session_state.current_analysis = calculate_tunnel_support(
                    st.session_state.analysis_params["depth"],
                    st.session_state.analysis_params["soil_density"],
                    st.session_state.analysis_params["k0"],
                    st.session_state.analysis_params["tunnel_diameter"]
                )
            elif analysis_type == "RMR Analysis":
                st.session_state.current_analysis = calculate_rmr(
                    st.session_state.analysis_params["ucs"],
                    st.session_state.analysis_params["rqd"],
                    st.session_state.analysis_params["spacing"],
                    st.session_state.analysis_params["condition"],
                    st.session_state.analysis_params["groundwater"],
                    st.session_state.analysis_params["orientation"]
                )
            elif analysis_type == "TBM Performance":
                st.session_state.current_analysis = estimate_tbm_performance(
                    st.session_state.analysis_params["ucs"],
                    st.session_state.analysis_params["rqd"],
                    st.session_state.analysis_params["joint_spacing"],
                    st.session_state.analysis_params["abrasivity"],
                    st.session_state.analysis_params["diameter"]
                )

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
                x=[0, st.session_state.analysis_params["tunnel_diameter"]],
                y=[st.session_state.analysis_params["depth"], 
                   st.session_state.analysis_params["depth"]],
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
        
        elif analysis_type == "RMR Analysis":
            ratings = st.session_state.current_analysis["ratings"]
            fig = go.Figure(data=[
                go.Bar(x=list(ratings.keys()), y=list(ratings.values()))
            ])
            fig.update_layout(
                title="RMR Component Ratings",
                xaxis_title="Parameters",
                yaxis_title="Rating"
            )
            st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")
