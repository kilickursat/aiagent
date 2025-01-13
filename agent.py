import math
import logging
import json
import pandas as pd
import numpy as np
import re
import requests
from typing import Dict, List
from datetime import datetime
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import (
    tool, 
    CodeAgent, 
    HfApiModel, 
    ManagedAgent, 
    ToolCallingAgent,
    DuckDuckGoSearchTool
)
from huggingface_hub import login
import plotly.graph_objects as go

# Initialize Hugging Face login
login('huggingface_api')  # Replace with your token

class Config:
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> dict:
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.default_config()

    def default_config(self) -> dict:
        return {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "tbm": {
                "efficiency_factor": 0.85,
                "cutter_life": 100,
                "maintenance_factor": 0.9
            },
            "face_stability": {
                "safety_factor": 1.5,
                "water_pressure_factor": 1.2
            },
            "visualization": {
                "colors": {
                    "soil": "#8B4513",
                    "rock": "#808080",
                    "water": "#4169E1"
                }
            }
        }

class LoggerSetup:
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.config['logging']['level']),
            format=self.config.config['logging']['format'],
            filename=f'geo_agent_{datetime.now().strftime("%Y%m%d")}.log'
        )
@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit and retrieve content from.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
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
    """Searches for geotechnical information using DuckDuckGo.

    Args:
        query: The search query for finding geotechnical information.

    Returns:
        Search results as formatted text.
    """
    search_tool = DuckDuckGoSearchTool()
    try:
        results = search_tool(query)  # Changed from .run() to direct call
        return str(results)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def classify_soil(soil_type: str, plasticity_index: float, liquid_limit: float) -> Dict:
    """Classify soil using USCS classification system.

    Args:
        soil_type: Type of soil (clay, sand, silt)
        plasticity_index: Plasticity index value
        liquid_limit: Liquid limit value

    Returns:
        Dictionary containing soil classification and description
    """
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
    """Calculate tunnel support pressure and related parameters.

    Args:
        depth: Tunnel depth from surface in meters
        soil_density: Soil density in kg/m³
        k0: At-rest earth pressure coefficient
        tunnel_diameter: Tunnel diameter in meters

    Returns:
        Dictionary containing support pressures, stresses and safety factors
    """
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
@tool
def calculate_rmr(ucs: float, rqd: float, spacing: float, condition: int, groundwater: int, orientation: int) -> Dict:
    """Calculate Rock Mass Rating (RMR) classification.

    Args:
        ucs: Uniaxial compressive strength in MPa
        rqd: Rock Quality Designation as percentage
        spacing: Joint spacing in meters
        condition: Joint condition rating (0-30)
        groundwater: Groundwater condition rating (0-15)
        orientation: Joint orientation rating (-12-0)

    Returns:
        Dictionary containing RMR value, rock class, and component ratings
    """
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
    """Calculate Q-system rating and support requirements.

    Args:
        rqd: Rock Quality Designation as percentage
        jn: Joint set number
        jr: Joint roughness number
        ja: Joint alteration number
        jw: Joint water reduction factor
        srf: Stress Reduction Factor

    Returns:
        Dictionary containing Q-value and support recommendations
    """
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
def estimate_tbm_performance(ucs: float, rqd: float, joint_spacing: float,
                           abrasivity: float, diameter: float) -> Dict:
    """Estimate TBM performance parameters.

    Args:
        ucs: Uniaxial compressive strength in MPa
        rqd: Rock Quality Designation as percentage
        joint_spacing: Average joint spacing in meters
        abrasivity: Cerchar abrasivity index
        diameter: TBM diameter in meters

    Returns:
        Dictionary containing TBM performance estimates
    """
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
@tool
def analyze_face_stability(depth: float, diameter: float, soil_density: float,
                         cohesion: float, friction_angle: float, water_table: float) -> Dict:
    """Analyze tunnel face stability.

    Args:
        depth: Tunnel depth in meters
        diameter: Tunnel diameter in meters
        soil_density: Soil density in kg/m³
        cohesion: Soil cohesion in kPa
        friction_angle: Soil friction angle in degrees
        water_table: Water table depth from surface in meters

    Returns:
        Dictionary containing stability analysis results
    """
    g = 9.81
    sigma_v = depth * soil_density * g / 1000
    water_pressure = (depth - water_table) * 9.81 if water_table < depth else 0
    N = (sigma_v - water_pressure) * math.tan(math.radians(friction_angle)) + cohesion
    fs = N / (0.5 * soil_density * g * diameter / 1000)

    return {
        "stability_ratio": round(N, 2),
        "factor_of_safety": round(fs, 2),
        "water_pressure": round(water_pressure, 2),
        "support_pressure_required": round(sigma_v/fs, 2) if fs < 1.5 else 0
    }

@tool
def import_borehole_data(file_path: str) -> Dict:
    """Import and process borehole data.

    Args:
        file_path: Path to borehole data CSV file

    Returns:
        Dictionary containing processed borehole data
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = ['depth', 'soil_type', 'N_value', 'moisture']

        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns in borehole data")

        return {
            "total_depth": df['depth'].max(),
            "soil_layers": df['soil_type'].nunique(),
            "ground_water_depth": df[df['moisture'] > 50]['depth'].min(),
            "average_N_value": df['N_value'].mean(),
            "soil_profile": df.groupby('soil_type')['depth'].agg(['min', 'max']).to_dict()
        }
    except Exception as e:
        logging.error(f"Error processing borehole data: {e}")
        raise

@tool
def visualize_3d_results(coordinates: str, geology_data: str, analysis_data: str) -> Dict:
    """Create 3D visualization of tunnel and analysis results.

    Args:
        coordinates: JSON string of tunnel coordinates [[x,y,z],...]
        geology_data: JSON string of geological layers
        analysis_data: JSON string of analysis results

    Returns:
        Dictionary containing plot data and statistics
    """
    tunnel_path = json.loads(coordinates)
    geology = json.loads(geology_data)
    analysis_results = json.loads(analysis_data)

    fig = go.Figure()
    x, y, z = zip(*tunnel_path)

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Tunnel Alignment'))
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=[r['factor_of_safety'] for r in analysis_results['stability']],
            colorscale='Viridis',
        ),
        name='Stability Analysis'
    ))

    return {
        "plot": fig.to_dict(),
        "statistics": {
            "tunnel_length": sum(math.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2)
                               for i in range(1, len(x))),
            "depth_range": [min(z), max(z)],
            "critical_sections": [i for i, r in enumerate(analysis_results['stability'])
                                if r['factor_of_safety'] < 1.5]
        }
    }
def get_support_recommendations(rmr: int) -> Dict:
    """Get support recommendations based on RMR value.

    Args:
        rmr: Rock Mass Rating value

    Returns:
        Dictionary containing support recommendations
    """
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

def get_q_support_category(q: float) -> Dict:
    """Get Q-system support recommendations.

    Args:
        q: Q-system value

    Returns:
        Dictionary containing support recommendations
    """
    if q > 40:
        return {
            "support_type": "No support required",
            "bolting": "None or occasional spot bolting",
            "shotcrete": "None required"
        }
    elif q > 10:
        return {
            "support_type": "Spot bolting",
            "bolting": "Spot bolts in crown, spaced 2.5m",
            "shotcrete": "None required"
        }
    elif q > 4:
        return {
            "support_type": "Systematic bolting",
            "bolting": "Systematic bolts in crown spaced 2m, occasional wire mesh",
            "shotcrete": "40-100mm where needed"
        }
    elif q > 1:
        return {
            "support_type": "Systematic bolting with shotcrete",
            "bolting": "Systematic bolts spaced 1-1.5m with wire mesh in crown and sides",
            "shotcrete": "50-90mm in crown and 30mm on sides"
        }
    else:
        return {
            "support_type": "Heavy support",
            "bolting": "Systematic bolts spaced 1m with wire mesh",
            "shotcrete": "90-120mm in crown and 100mm on sides",
            "additional": "Consider steel ribs, forepoling, or face support"
        }

# Initialize model and agents
model = HfApiModel("mistralai/Mistral-Nemo-Instruct-2407")

# Create web search agent
web_agent = ToolCallingAgent(
    tools=[search_geotechnical_data, visit_webpage],
    model=model,
    max_steps=10
)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="geotech_web_search",
    description="Performs web searches for geotechnical data and case studies."
)

# Create geotechnical calculation agent
geotech_agent = ToolCallingAgent(
    tools=[
        classify_soil,
        calculate_tunnel_support,
        calculate_rmr,
        calculate_q_system,
        estimate_tbm_performance,
        analyze_face_stability,
        import_borehole_data,
        visualize_3d_results
    ],
    model=model,
    max_steps=10
)

managed_geotech_agent = ManagedAgent(
    agent=geotech_agent,
    name="geotech_analysis",
    description="Performs geotechnical calculations and analysis."
)

# Create manager agent
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent, managed_geotech_agent],
    additional_authorized_imports=["time", "numpy", "pandas"]
)
# Fix managed agent calls
def process_request(agent: ManagedAgent, request: str):
    """Helper function to process managed agent requests.

    Args:
        agent: ManagedAgent instance
        request: Query string
    """
    return agent(request=request)
# Main execution
if __name__ == "__main__":
    config = Config()
    logger = LoggerSetup(config)
    
    try:
        query = "Analyze tunnel stability: depth=30m, soil_type=clay, diameter=6m"
        
        # Convert results to strings before passing to manager
        web_result = str(process_request(managed_web_agent, query))
        geotech_result = str(process_request(managed_geotech_agent, query))
        
        final_result = manager_agent.run(
            f"Web Search Results: {web_result}\n"
            f"Technical Analysis: {geotech_result}\n"
            f"Query: {query}"
        )
        print(f"Analysis Result:\n{final_result}")

    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        print(f"An error occurred: {e}")
