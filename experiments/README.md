# Coffee Data Analysis and Experiments

This directory contains Python scripts for analyzing coffee brewing and grinding data, as well as running machine learning experiments using XGBoost and CatBoost.

## Contents

- `coffee_ml_analysis.py` - General coffee data analysis script
- `brew_analysis.py` - Focused analysis of brew data
- `stress_coffee_recommendation.py` - Proof of concept for stress-based coffee recommendations
- `system_diagram.py` - Generates a diagram of the stress-based coffee system
- `muse_stress_coffee.py` - Integration with Muse 2 headband for real EEG-based stress detection
- `requirements.txt` - List of required Python packages

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

For Muse headband integration, you'll need additional dependencies:

```bash
pip install muselsl pylsl
```

## Running Experiments

### General Coffee Data Analysis

```bash
python coffee_ml_analysis.py
```

This script:
- Loads and explores both coffee grinding and brewing data
- Performs correlation analysis
- Trains XGBoost and CatBoost models
- Compares model performance
- Generates visualizations for feature importance

### Brew Data Analysis

```bash
python brew_analysis.py
```

This script focuses specifically on brewing data:
- Performs detailed exploratory data analysis
- Conducts correlation analysis
- Builds prediction models for extraction yield

### Stress-Based Coffee Recommendation System

```bash
python stress_coffee_recommendation.py
```

This proof of concept:
- Simulates EEG data collection from a Muse 2 headband
- Analyzes stress levels
- Recommends coffee parameters (dose, resting time, blends)
- Demonstrates interface with coffee machine via Raspberry Pi

### System Diagram Generation

```bash
python system_diagram.py
```

Generates a visual diagram of the stress-based coffee recommendation system architecture.

### Muse Headband Integration

```bash
python muse_stress_coffee.py --use-real-muse
```

This experimental script:
- Integrates with real Muse 2 headband via [muse-lsl](https://github.com/alexandrebarachant/muse-lsl)
- Collects and processes EEG data for stress analysis
- Uses alpha/beta ratio and other EEG metrics for scientifically-based stress calculation
- Generates coffee recommendations based on real stress measurements

Command line options:
- `--use-real-muse`: Use real Muse headband (if available) instead of simulated data
- `--measurements`: Number of stress measurements to take (default: 1)
- `--measurement-duration`: Duration of each measurement in seconds (default: 10)
- `--interval`: Time between measurements in seconds (default: 60)

## Output

The scripts generate various outputs:
- Feature importance plots
- Correlation matrices
- Model comparison graphs
- Stress history charts (for stress-based recommendation scripts)
- Coffee recommendations in JSON format (in the `output` directory)

## Data

The scripts expect the following data files:
- `../data/nlc-grind-data.csv` - Coffee grinder data
- `../data/nlc-brew-data.csv` - Coffee brewing data 