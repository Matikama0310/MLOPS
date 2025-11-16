# 0-try: Experimentation

This folder contains the exploratory phase used to select the best model configuration for predicting mental health treatment.

## Contents
- `experiment.ipynb`: Notebook that loads the Kaggle *OSMI Mental Health in Tech* survey, performs cleaning/feature selection, and runs model comparisons. It writes the chosen hyperparameters and feature lists to `best_config.json` for downstream training.
- `create_experiment_notebook.py`: Utility that programmatically generates the experiment notebook template.

## How to use
1. Open and run `experiment.ipynb` end-to-end to evaluate candidate models.
2. After the notebook finishes, ensure `best_config.json` is saved at the repository root (consumed by training scripts in later stages).
3. If the notebook is missing, run `python create_experiment_notebook.py` to regenerate it before executing the cells.
