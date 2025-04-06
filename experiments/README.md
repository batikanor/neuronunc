# Coffee Data Analysis and ML Experiments

This directory contains Python scripts for analyzing coffee brewing and grinding data, and running machine learning experiments using XGBoost and CatBoost.

## Files

- `coffee_ml_analysis.py`: Main script for analyzing both grind and brew data
- `brew_analysis.py`: Script focused specifically on brew data analysis
- `requirements.txt`: Python dependencies required to run the scripts

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Experiments

### 1. General Coffee Data Analysis

Run the main analysis script:

```bash
python coffee_ml_analysis.py
```

This script:
- Loads both grind and brew data
- Performs exploratory data analysis
- Conducts correlation analysis
- Trains and evaluates XGBoost and CatBoost models
- Saves visualization and analysis results

### 2. Focused Brew Data Analysis

Run the brew-specific analysis:

```bash
python brew_analysis.py
```

This script:
- Loads and analyzes just the brew data (with sampling for large files)
- Performs detailed correlation analysis
- Trains models with multiple target variables (extractionYield, TDS, etc.)
- Compares model performance and analyzes feature importance

## Output

The scripts generate several output files:
- Feature importance plots
- Correlation matrices
- Model comparison graphs
- Detailed data analysis text files

All visualization images are saved in a `figures` directory that will be created automatically.

## Data Structure

These scripts analyze coffee-related CSV files located in the parent `data` directory:
- `../data/nlc-grind-data.csv`: Data about coffee grinding events
- `../data/nlc-brew-data.csv`: Data about coffee brewing events 