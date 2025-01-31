import streamlit as st
import math
import logging
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import requests
from datetime import datetime
from huggingface_hub import login
from typing import Dict, List
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool, CodeAgent, HfApiModel, ManagedAgent, ToolCallingAgent, DuckDuckGoSearchTool
import traceback

# Page configuration
st.set_page_config(
    page_title="Advanced Geotechnical AI Agent",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_params' not in st.session_state:
    st.session_state.analysis_params = {}

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"Error fetching webpage: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

@tool
def search_geotechnical_data(query: str) -> str:
    """Searches for geotechnical information using DuckDuckGo."""
    search_tool = DuckDuckGoSearchTool()
    try:
        filtered_query = (
            f"{query} site:.edu OR site:.org OR site:.gov OR site:.com "
            f'"geotechnical engineering" OR "tunnelling" OR "tunnel boring machine" '
            f'OR "underground space technology" OR "excavation engineering" '
            f'OR "mechanical excavation" OR "geology" OR "mining" OR "mining engineering"'
        )
        results = search_tool(filtered_query)
        return str(results)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def classify_soil(soil_type: str, plasticity_index: float, liquid_limit: float) -> Dict:
    """Classify soil using USCS classification system."""
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
    """Calculate tunnel support pressure and related parameters."""
    g = 9.81
    vertical_stress = depth * soil_density * g / 1000
    horizontal_stress = k0 * vertical_stress
    support_pressure = (vertical_stress + horizontal_stress) / 2
    safety_factor = 1.5 if depth < 30 else 2.0
    return {
        "support_pressure": support_pressure,
        "design_pressure": support_pressure * safety_factor,
        "safety_factor": safety_factor,
        "vertical_stress": vertical_stress,
        "horizontal_stress": horizontal_stress
    }

def get_support_recommendations(rmr: int) -> Dict:
    """Get support recommendations based on RMR value."""
    if rmr > 80:
        return {
            "excavation": "Full face, 3m advance",
            "support": "Generally no support required",
            "bolting": "Spot bolting if needed",
            "shotcrete": "None required",
            "steel_sets": "None required"
        }
    elif rmr > 60:
        return {
            "excavation": "Full face, 1.0-1.5m advance",
            "support": "Complete within 20m of face",
            "bolting": "Systematic bolting, 4m length, spaced 1.5-2m",
            "shotcrete": "50mm in crown where required",
            "steel_sets": "None required"
        }
    elif rmr > 40:
        return {
            "excavation": "Top heading and bench, 1.5-3m advance",
            "support": "Complete within 10m of face",
            "bolting": "Systematic bolting, 4-5m length, spaced 1-1.5m",
            "shotcrete": "50-100mm in crown and 30mm in sides",
            "steel_sets": "Light to medium ribs spaced 1.5m where required"
        }
    else:
        return {
            "excavation": "Multiple drifts, 0.5-1.5m advance",
            "support": "Install support concurrent with excavation",
            "bolting": "Systematic bolting with shotcrete and steel sets",
            "shotcrete": "100-150mm in crown and sides",
            "steel_sets": "Medium to heavy ribs spaced 0.75m"
        }

@tool
def calculate_rmr(ucs: float, rqd: float, spacing: float, condition: int, groundwater: int, orientation: int) -> Dict:
    """Calculate Rock Mass Rating (RMR) classification."""
    if ucs > 250: ucs_rating = 15
    elif ucs > 100: ucs_rating = 12
    elif ucs > 50: ucs_rating = 7
    elif ucs > 25: ucs_rating = 4
    else: ucs_rating = 2
    if rqd > 90: rqd_rating = 20
    elif rqd > 75: rqd_rating = 17
    elif rqd > 50: rqd_rating = 13
    elif rqd > 25: rqd_rating = 8
    else: rqd_rating = 3
    if spacing > 2: spacing_rating = 20
    elif spacing > 0.6: spacing_rating = 15
    elif spacing > 0.2: spacing_rating = 10
    elif spacing > 0.06: spacing_rating = 8
    else: spacing_rating = 5
    total_rmr = ucs_rating + rqd_rating + spacing_rating + condition + groundwater + orientation
    if total_rmr > 80: rock_class = "I - Very good rock"
    elif total_rmr > 60: rock_class = "II - Good rock"
    elif total_rmr > 40: rock_class = "III - Fair rock"
    elif total_rmr > 20: rock_class = "IV - Poor rock"
    else: rock_class = "V - Very poor rock"
    return {
        "rmr_value": total_rmr,
        "rock_class": rock_class,
        "support_recommendations": get_support_recommendations(total_rmr),
        "component_ratings": {
            "ucs_rating": ucs_rating,
            "rqd_rating": rqd_rating,
            "spacing_rating": spacing_rating,
            "condition_rating": condition,
            "groundwater_rating": groundwater,
            "orientation_rating": orientation
        }
    }

@tool
def calculate_q_system(rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float) -> Dict:
    """Calculate Q-system rating and support requirements."""
    q_value = (rqd/jn) * (jr/ja) * (jw/srf)
    if q_value > 40: quality = "Exceptionally Good"
    elif q_value > 10: quality = "Very Good"
    elif q_value > 4: quality = "Good"
    elif q_value > 1: quality = "Fair"
    elif q_value > 0.1: quality = "Poor"
    else: quality = "Extremely Poor"
    return {
        "q_value": round(q_value, 2),
        "rock_quality": quality,
        "support_category": get_q_support_category(q_value),
        "parameters": {
            "RQD/Jn": round(rqd/jn, 2),
            "Jr/Ja": round(jr/ja, 2),
            "Jw/SRF": round(jw/srf, 2)
        }
    }

@tool
def estimate_tbm_performance(ucs: float, rqd: float, joint_spacing: float, abrasivity: float, diameter: float) -> Dict:
    """Estimate TBM performance parameters."""
    pr = 20 * (1/ucs) * (rqd/100) * (1/abrasivity)
    utilization = 0.85 - (0.01 * (abrasivity/2))
    advance_rate = pr * utilization * 24
    cutter_life = 100 * (250/ucs) * (2/abrasivity)
    return {
        "penetration_rate": round(pr, 2),
        "daily_advance": round(advance_rate, 2),
        "utilization": round(utilization * 100, 1),
        "cutter_life_hours": round(cutter_life, 0),
        "estimated_completion_days": round(1000/advance_rate, 0)
    }

def generate_detailed_response(query: str, web_results: str) -> str:
    """
    Generates a detailed response based on the query and web search results.
    Includes citations, equations, and longer contextual explanations.
    """
    response = (
        f"Based on recent research and industry standards in geotechnical engineering, tunnelling, "
        f"and related fields, here is the information regarding your query:\n\n"
    )
    
    if "specific energy" in query.lower():
        response += (
            "Specific energy is a critical parameter in tunnel boring machine (TBM) operations. "
            "It is defined as the amount of energy required to excavate a unit volume of rock or soil.\n\n"
        )
        # Add equation for specific energy
        response += (
            "The specific energy (\(SE\)) can be calculated using the following equation:\n\n"
            "\\[ SE = \\frac{P}{A \\cdot V} \\]\n\n"
            "Where:\n"
            "- \(P\) is the power consumed by the TBM cutterhead (in watts),\n"
            "- \(A\) is the cross-sectional area of the tunnel (in square meters),\n"
            "- \(V\) is the penetration rate (in meters per second).\n\n"
        )
        response += (
            "For example, if a TBM consumes 1000 kW of power, has a tunnel cross-sectional area of 50 m², "
            "and achieves a penetration rate of 0.01 m/s, the specific energy would be:\n\n"
            "\\[ SE = \\frac{1000000}{50 \\cdot 0.01} = 200000 \\, \\text{J/m³} \\]\n\n"
        )
        response += (
            "This value helps engineers assess the performance of the TBM and optimize excavation parameters.\n\n"
        )
    elif "rock mass rating" in query.lower():
        response += (
            "The Rock Mass Rating (RMR) system is widely used to classify the quality of rock masses. "
            "It considers parameters such as uniaxial compressive strength (UCS), Rock Quality Designation (RQD), "
            "joint spacing, joint condition, groundwater conditions, and joint orientation.\n\n"
        )
        # Add equation for RMR calculation
        response += (
            "The total RMR value is calculated as the sum of individual ratings for each parameter:\n\n"
            "\\[ \\text{RMR} = \\text{UCS Rating} + \\text{RQD Rating} + \\text{Spacing Rating} + \\text{Condition Rating} "
            "+ \\text{Groundwater Rating} + \\text{Orientation Rating} \\]\n\n"
        )
        response += (
            "For example, a rock mass with high UCS, excellent RQD, wide joint spacing, and favorable orientation "
            "would result in a high RMR value, indicating very good rock quality.\n\n"
        )
    elif "q-system" in query.lower():
        response += (
            "The Q-system is another widely used method for rock mass classification. It is calculated using the following formula:\n\n"
            "\\[ Q = \\left(\\frac{RQD}{J_n}\\right) \\cdot \\left(\\frac{J_r}{J_a}\\right) \\cdot \\left(\\frac{J_w}{SRF}\\right) \\]\n\n"
            "Where:\n"
            "- \(RQD\) is the Rock Quality Designation,\n"
            "- \(J_n\) is the joint set number,\n"
            "- \(J_r\) is the joint roughness number,\n"
            "- \(J_a\) is the joint alteration number,\n"
            "- \(J_w\) is the joint water reduction factor,\n"
            "- \(SRF\) is the stress reduction factor.\n\n"
        )
        response += (
            "The Q-value is then used to determine the support requirements for tunnels and underground excavations.\n\n"
        )
    elif "tunnel boring machine" in query.lower() or "tbm" in query.lower():
        response += (
            "Tunnel Boring Machines (TBMs) are advanced mechanical systems used for excavating tunnels through various types of ground, "
            "including rock, soil, and mixed-face conditions. Key performance metrics for TBMs include penetration rate, advance rate, "
            "utilization, and cutter life.\n\n"
        )
        # Add equation for penetration rate
        response += (
            "The penetration rate (\(PR\)) of a TBM can be estimated using empirical models, such as:\n\n"
            "\\[ PR = C_1 \\cdot \\frac{RQD}{UCS} \\cdot \\frac{1}{CAI} \\]\n\n"
            "Where:\n"
            "- \(C_1\) is an empirical constant,\n"
            "- \(RQD\) is the Rock Quality Designation,\n"
            "- \(UCS\) is the Uniaxial Compressive Strength of the rock,\n"
            "- \(CAI\) is the Cerchar Abrasivity Index.\n\n"
        )
        response += (
            "For example, a TBM operating in rock with \(RQD = 75\), \(UCS = 100 \, \\text{MPa}\), and \(CAI = 2\) "
            "might achieve a penetration rate of:\n\n"
            "\\[ PR = 20 \\cdot \\frac{75}{100} \\cdot \\frac{1}{2} = 7.5 \\, \\text{mm/rev} \\]\n\n"
        )
    elif "groundwater" in query.lower():
        response += (
            "Groundwater conditions play a critical role in geotechnical engineering, particularly in tunnelling and excavation projects. "
            "High groundwater pressures can lead to instability, increased excavation costs, and safety risks.\n\n"
        )
        # Add equation for groundwater pressure
        response += (
            "The groundwater pressure (\(P_w\)) at a given depth (\(z\)) can be calculated using the hydrostatic pressure equation:\n\n"
            "\\[ P_w = \\gamma_w \\cdot z \\]\n\n"
            "Where:\n"
            "- \(\\gamma_w\) is the unit weight of water (\(9.81 \\, \\text{kN/m³}\)),\n"
            "- \(z\) is the depth below the water table (in meters).\n\n"
        )
        response += (
            "For example, at a depth of 20 meters below the water table, the groundwater pressure would be:\n\n"
            "\\[ P_w = 9.81 \\cdot 20 = 196.2 \\, \\text{kPa} \\]\n\n"
        )
    else:
        # General response for other queries
        response += f"{web_results}\n\n"
    
    # Append references
    response += (
        "**References:**\n"
        "[1] Identification and optimization of energy consumption by shield machines <button class="citation-flag" data-index="6">.\n"
        "[2] Estimating TBM Specific Energy by Employing the Rock Mass Parameters <button class="citation-flag" data-index="2">.\n"
        "[3] Estimation of the specific energy of tunnel boring machines <button class="citation-flag" data-index="3">.\n"
        "[4] Use of specific drilling energy for rock mass characterization <button class="citation-flag" data-index="4">.\n"
        "[5] Combining the RMR, Q and RMi classification systems <button class="citation-flag" data-index="7">.\n"
        "[6] Geotechnical Challenges in Tunnelling Projects <button class="citation-flag" data-index="10">.\n"
    )
    
    return response

def process_request(request: str):
    """Processes the user's request and generates a detailed response."""
    try:
        web_result = search_geotechnical_data(request)
        detailed_response = generate_detailed_response(request, web_result)
        return detailed_response
    except Exception as e:
        return f"Error: {str(e)}\nFull traceback:\n{traceback.format_exc()}"

# Initialize agent
managed_web_agent, managed_geotech_agent, manager_agent = initialize_agents()

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
st.title("🏗️ Geotechnical AI Agent by Qwen2.5-Coder-32B-Instruct")
st.subheader("💬 Chat Interface")
user_input = st.text_input("Ask a question:")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Processing..."):
        response = process_request(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Update chat display section
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        try:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            content = msg["content"]
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2)
            st.markdown(f"{role_icon} **{msg['role'].title()}:** {content}", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying message: {str(e)}")

# Analysis Results Section
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
        if "component_ratings" in st.session_state.current_analysis:
            ratings = st.session_state.current_analysis["component_ratings"]
            labels = {
                "ucs_rating": "UCS",
                "rqd_rating": "RQD",
                "spacing_rating": "Spacing",
                "condition_rating": "Condition",
                "groundwater_rating": "Groundwater",
                "orientation_rating": "Orientation"
            }
            fig = go.Figure(data=[
                go.Bar(x=[labels[k] for k in ratings.keys()], 
                       y=list(ratings.values()))
            ])
            fig.update_layout(
                title="RMR Component Ratings",
                xaxis_title="Parameters",
                yaxis_title="Rating"
            )
            st.plotly_chart(fig)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ by Kilic Intelligence")
