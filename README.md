# AI-Driven Geotechnical Analysis Agent

Welcome to the **AI-Driven Geotechnical Analysis Agent** repository! This project provides tools and agent-based functionality for analyzing geotechnical data, visualizing results, and estimating tunneling-related metrics using AI models and integrated tools. Designed for engineers, researchers, and enthusiasts in geotechnical engineering and tunneling, this solution streamlines complex calculations and enhances decision-making processes.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Tools and Agents](#tools-and-agents)
6. [Example](#example)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **Geotechnical Analysis Tools**:
  - Soil classification using the USCS classification system.
  - Tunnel support pressure and safety factor calculation.
  - Rock Mass Rating (RMR) and Q-System analysis for rock support classification.
  - TBM performance estimation, including penetration rates, cutter life, and advance rates.
  - Tunnel face stability analysis based on geotechnical parameters.
  - Borehole data processing for detailed soil and water table analysis.

- **Web Search and Information Extraction**:
  - Integrated DuckDuckGo search for finding geotechnical data and case studies.
  - Webpage content extraction and markdown formatting.

- **Visualization**:
  - 3D visualizations of tunnel paths and stability factors using Plotly.

- **AI-Driven Operations**:
  - Managed agents for executing web searches and technical calculations.
  - Modular and extensible architecture leveraging **smolagents** and the Hugging Face ecosystem.

---

## Installation

### Prerequisites
- Python 3.8+
- Dependencies listed in `requirements.txt`

### Steps
1. Clone this repository:
    ```bash
    git clone https://github.com/kilickursat/aiagent.git
    cd aiagent
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Set up Hugging Face API key:
    - Sign up at [Hugging Face](https://huggingface.co/) if you don’t have an account.
    - Get your API token [here](https://huggingface.co/settings/tokens).
    - Add your token in the script where prompted or through environment variables.

---

## Configuration

The project reads settings from a `config.json` file. If the file is missing, it uses default settings. You can customize:
- Logging preferences.
- Parameters for TBM efficiency, cutter life, and maintenance factors.
- Visualization colors for soil, rock, and water.
- Safety factors and water pressure adjustments.

Example `config.json`:
```json
{
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
```

---

## Usage

### Running the Application
To execute the main analysis pipeline:
```bash
python main.py
```

### Agents and Tools
- **Web Search**: Provides information about geotechnical data using a search query.
- **Geotechnical Calculations**: Performs detailed analyses like tunnel support requirements and soil classification.

---

## Tools and Agents

### Tools
- **`visit_webpage(url: str)`**: Extracts content from a webpage as markdown.
- **`search_geotechnical_data(query: str)`**: Searches for geotechnical data and case studies online.
- **`classify_soil(...)`**: Classifies soil using USCS.
- **`calculate_tunnel_support(...)`**: Analyzes support pressures and stresses for tunnels.
- **`calculate_rmr(...)`**: Evaluates rock quality using the RMR method.
- **`calculate_q_system(...)`**: Calculates rock quality and support needs using the Q-System.
- **`estimate_tbm_performance(...)`**: Estimates TBM performance metrics.
- **`analyze_face_stability(...)`**: Examines tunnel face stability parameters.
- **`import_borehole_data(file_path: str)`**: Processes borehole data from a CSV.
- **`visualize_3d_results(...)`**: Produces 3D tunnel path and stability visualizations.

### Agents
- **`web_agent`**: Performs web-based queries using DuckDuckGo and returns relevant results.
- **`geotech_agent`**: Executes geotechnical calculations using the defined tools.

---

## Example

### Sample Query
Analyze tunnel stability:
```python
query = "Analyze tunnel stability: depth=30m, soil_type=clay, diameter=6m"
web_result = process_request(managed_web_agent, query)
geotech_result = process_request(managed_geotech_agent, query)

final_result = manager_agent.run({
    "web_data": web_result,
    "technical_analysis": geotech_result,
    "query": query
})
print(final_result)
```

---

## Contributing

We welcome contributions to improve this project! Here's how to get started:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -am 'Add feature name'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


