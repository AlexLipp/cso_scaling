# Fitting heavy tailed distributions to CSO spill durations 

This repository contains a demonstration of the heavy tailed distribution fitting described in the manuscript '_Groundwater infiltration drives heavy-tailed distributions of Combined Sewer Overflow durations: insights from Event Duration Monitoring_' by Alex Lipp & Barnaby Dobson 

## Installation

### Prerequisites

- Python 3.8 or higher

### Setup

1. **Clone or download this repository** to your local machine

2. **Navigate to the project directory** in your terminal:
   ```bash
   cd /path/to/cso_scaling
   ```

3. **Create and activate a virtual environment** (recommended):

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages 

- **matplotlib** - for plotting and visualization
- **numpy** - for numerical computations
- **pandas** - for data manipulation and analysis
- [**powerlaw**](https://pypi.org/project/powerlaw/) - for fitting power law distributions

## Usage

With the virtual environment activated, you can run the example script:

```bash
python fit_distribution.py
```
This fits heavy tailed distributions, including the stretched exponential to the 2023 Event Duration Monitoring data from Southern Water. 
