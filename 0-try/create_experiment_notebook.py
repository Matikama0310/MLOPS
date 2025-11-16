#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create experiment.ipynb notebook.

Usage:
    python create_experiment_notebook.py
    
Output:
    experiment.ipynb
"""

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "id": "header",
            "metadata": {},
            "source": [
                "# Mental Health Model Experimentation\n",
                "\n",
                "**Location:** `0 try/experiment.ipynb`\n",
                "\n",
                "**Outputs to:** `../3-cicd/`\n",
                "\n",
                "**Goal:** Find the best model, hyperparameters, and features\n",
                "\n",
                "**Output:** `best_config.json` in `3-cicd/` folder for `train.py`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "setup",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Setup and imports\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import json\n",
                "import re\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.metrics import make_scorer, fbeta_score\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from xgboost import XGBClassifier\n",
                "from sklearn.inspection import permutation_importance\n",
                "\n",
                "pd.set_option('display.max_columns', None)\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "\n",
                "OUTPUT_DIR = Path('../3-cicd')\n",
                "OUTPUT_DIR.mkdir(exist_ok=True)\n",
                "\n",
                "print('✓ Libraries imported')\n",
                "print(f'✓ Output: {OUTPUT_DIR.absolute()}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "load",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load data\n",
                "df = pd.read_csv('../data/raw/survey.csv')\n",
                "print(f'Dataset: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "preprocess",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Preprocessing\n",
                "def clean_gender(gen):\n",
                "    s = str(gen).strip().lower()\n",
                "    s = re.sub(r'[\\W_]+', ' ', s).strip()\n",
                "    if s in {'m','male','man','make','mal','malr','msle','masc','mail','boy'}:\n",
                "        return 'Male'\n",
                "    if s in {'f','female','woman','femake','femail','femme','girl'}:\n",
                "        return 'Female'\n",
                "    return 'Other'\n",
                "\n",
                "target_col = 'treatment'\n",
                "features_to_drop = ['Timestamp','Country','state','comments',target_col]\n",
                "\n",
                "y = df[target_col].map({'Yes':1,'No':0}).astype(int)\n",
                "X = df.drop(columns=features_to_drop)\n",
                "if 'Gender' in X.columns:\n",
                "    X['Gender'] = X['Gender'].apply(clean_gender)\n",
                "\n",
                "print(f'Features: {X.shape[1]}')\n",
                "print(f'Samples: {X.shape[0]}')\n",
                "print('\\nTarget distribution:')\n",
                "print(y.value_counts(normalize=True))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "models",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define models\n",
                "models_to_test = {\n",
                "    'LogisticRegression': {\n",
                "        'model': LogisticRegression(solver='liblinear',max_iter=5000,random_state=42),\n",
                "        'params': {'C':1.0,'penalty':'l2'}\n",
                "    },\n",
                "    'RandomForest': {\n",
                "        'model': RandomForestClassifier(random_state=42),\n",
                "        'params': {'n_estimators':300,'max_depth':10,'min_samples_split':2,'class_weight':'balanced'}\n",
                "    },\n",
                "    'XGBoost': {\n",
                "        'model': XGBClassifier(random_state=42),\n",
                "        'params': {'n_estimators':400,'max_depth':6,'learning_rate':0.01,'subsample':0.8,'colsample_bytree':1.0,'scale_pos_weight':1.26}\n",
                "    }\n",
                "}\n",
                "\n",
                "print('Models to test:')\n",
                "for name in models_to_test:\n",
                "    print(f'  - {name}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "preprocessor",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Build preprocessor\n",
                "def build_preprocessor(X):\n",
                "    num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()\n",
                "    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()\n",
                "    num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])\n",
                "    cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])\n",
                "    return ColumnTransformer([('num',num_pipe,num_cols),('cat',cat_pipe,cat_cols)],remainder='drop')\n",
                "\n",
                "preprocessor = build_preprocessor(X)\n",
                "print('✓ Preprocessor ready')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "cv",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Cross-validation\n",
                "print('='*70)\n",
                "print('CROSS-VALIDATION')\n",
                "print('='*70)\n",
                "\n",
                "cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)\n",
                "scoring = {'recall':'recall','precision':'precision','f1':'f1','roc_auc':'roc_auc','fbeta':make_scorer(fbeta_score,beta=1.5)}\n",
                "\n",
                "results = []\n",
                "for name,config in models_to_test.items():\n",
                "    print(f'\\n🔄 {name}...')\n",
                "    model = config['model']\n",
                "    model.set_params(**config['params'])\n",
                "    pipe = Pipeline([('preprocess',preprocessor),('model',model)])\n",
                "    cv_results = cross_validate(pipe,X,y,cv=cv,scoring=scoring,n_jobs=-1,return_train_score=False)\n",
                "    result = {\n",
                "        'model':name,\n",
                "        'recall_mean':cv_results['test_recall'].mean(),\n",
                "        'recall_std':cv_results['test_recall'].std(),\n",
                "        'precision_mean':cv_results['test_precision'].mean(),\n",
                "        'precision_std':cv_results['test_precision'].std(),\n",
                "        'f1_mean':cv_results['test_f1'].mean(),\n",
                "        'f1_std':cv_results['test_f1'].std(),\n",
                "        'fbeta_mean':cv_results['test_fbeta'].mean(),\n",
                "        'fbeta_std':cv_results['test_fbeta'].std(),\n",
                "        'roc_auc_mean':cv_results['test_roc_auc'].mean(),\n",
                "        'roc_auc_std':cv_results['test_roc_auc'].std()\n",
                "    }\n",
                "    results.append(result)\n",
                "    print(f\"  Recall:    {result['recall_mean']:.3f} ± {result['recall_std']:.3f}\")\n",
                "    print(f\"  Precision: {result['precision_mean']:.3f} ± {result['precision_std']:.3f}\")\n",
                "    print(f\"  F-beta:    {result['fbeta_mean']:.3f} ± {result['fbeta_std']:.3f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "results",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Results\n",
                "results_df = pd.DataFrame(results).sort_values('fbeta_mean',ascending=False)\n",
                "print('\\n'+'='*70)\n",
                "print('RESULTS')\n",
                "print('='*70)\n",
                "print(results_df[['model','recall_mean','precision_mean','fbeta_mean']].to_string(index=False))\n",
                "best_model_name = results_df.iloc[0]['model']\n",
                "print(f'\\n🏆 Best: {best_model_name}')\n",
                "\n",
                "# Plot\n",
                "fig,ax = plt.subplots(1,2,figsize=(14,5))\n",
                "x = np.arange(len(results_df))\n",
                "ax[0].bar(x,results_df['recall_mean'],label='Recall')\n",
                "ax[0].bar(x,results_df['precision_mean'],alpha=0.7,label='Precision')\n",
                "ax[0].set_xticks(x)\n",
                "ax[0].set_xticklabels(results_df['model'])\n",
                "ax[0].legend()\n",
                "ax[0].set_title('Metrics')\n",
                "ax[1].scatter(results_df['precision_mean'],results_df['recall_mean'])\n",
                "for _,row in results_df.iterrows():\n",
                "    ax[1].annotate(row['model'],(row['precision_mean'],row['recall_mean']))\n",
                "ax[1].set_xlabel('Precision')\n",
                "ax[1].set_ylabel('Recall')\n",
                "plt.tight_layout()\n",
                "plt.savefig(OUTPUT_DIR/'model_comparison.png',dpi=150)\n",
                "plt.show()\n",
                "print(f'✓ Saved: {OUTPUT_DIR}/model_comparison.png')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "importance",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature importance\n",
                "print('='*70)\n",
                "print('FEATURE SELECTION')\n",
                "print('='*70)\n",
                "\n",
                "best_config = models_to_test[best_model_name]\n",
                "best_model = best_config['model']\n",
                "best_model.set_params(**best_config['params'])\n",
                "\n",
                "X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)\n",
                "pipe = Pipeline([('preprocess',preprocessor),('model',best_model)])\n",
                "pipe.fit(X_train,y_train)\n",
                "\n",
                "perm = permutation_importance(pipe.named_steps['model'],pipe.named_steps['preprocess'].transform(X_val),y_val,n_repeats=10,random_state=42,n_jobs=-1)\n",
                "\n",
                "try:\n",
                "    feat_names = pipe.named_steps['preprocess'].get_feature_names_out()\n",
                "except:\n",
                "    feat_names = [f'f_{i}' for i in range(len(perm.importances_mean))]\n",
                "\n",
                "orig_imp = {}\n",
                "for col in X.columns:\n",
                "    imp = [perm.importances_mean[i] for i,f in enumerate(feat_names) if col.lower() in f.lower()]\n",
                "    orig_imp[col] = max(imp) if imp else 0.0\n",
                "\n",
                "imp_df = pd.DataFrame([{'feature':k,'importance':v} for k,v in orig_imp.items()]).sort_values('importance',ascending=False)\n",
                "print('\\nTop features:')\n",
                "print(imp_df.head(10).to_string(index=False))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "select",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Feature selection\n",
                "thresh = 0.001\n",
                "selected = imp_df[imp_df['importance']>thresh]['feature'].tolist()\n",
                "removed = imp_df[imp_df['importance']<=thresh]['feature'].tolist()\n",
                "\n",
                "print(f'\\nThreshold: {thresh}')\n",
                "print(f'Selected: {len(selected)}')\n",
                "print(f'Removed: {len(removed)}')\n",
                "if removed:\n",
                "    print(f'\\nRemoved: {removed}')\n",
                "\n",
                "plt.figure(figsize=(10,6))\n",
                "plt.barh(imp_df['feature'],imp_df['importance'])\n",
                "plt.axvline(thresh,color='red',linestyle='--',label=f'Threshold={thresh}')\n",
                "plt.xlabel('Importance')\n",
                "plt.legend()\n",
                "plt.tight_layout()\n",
                "plt.savefig(OUTPUT_DIR/'feature_importance.png',dpi=150)\n",
                "plt.show()\n",
                "print(f'✓ Saved: {OUTPUT_DIR}/feature_importance.png')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "final",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Final evaluation\n",
                "X_sel = X[selected].copy()\n",
                "pipe_final = Pipeline([('preprocess',build_preprocessor(X_sel)),('model',best_model)])\n",
                "cv_final = cross_validate(pipe_final,X_sel,y,cv=cv,scoring=scoring,n_jobs=-1)\n",
                "\n",
                "print('\\nFinal CV:')\n",
                "print(f\"  Recall:    {cv_final['test_recall'].mean():.3f}\")\n",
                "print(f\"  Precision: {cv_final['test_precision'].mean():.3f}\")\n",
                "print(f\"  F-beta:    {cv_final['test_fbeta'].mean():.3f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "save",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Save config\n",
                "config = {\n",
                "    'model':{'name':best_model_name,'class':best_model_name,'hyperparameters':best_config['params']},\n",
                "    'features':{'selected':selected,'removed':removed,'importance_threshold':thresh},\n",
                "    'performance':{\n",
                "        'cv_recall_mean':float(cv_final['test_recall'].mean()),\n",
                "        'cv_recall_std':float(cv_final['test_recall'].std()),\n",
                "        'cv_precision_mean':float(cv_final['test_precision'].mean()),\n",
                "        'cv_precision_std':float(cv_final['test_precision'].std()),\n",
                "        'cv_f1_mean':float(cv_final['test_f1'].mean()),\n",
                "        'cv_f1_std':float(cv_final['test_f1'].std()),\n",
                "        'cv_fbeta_mean':float(cv_final['test_fbeta'].mean()),\n",
                "        'cv_fbeta_std':float(cv_final['test_fbeta'].std()),\n",
                "        'cv_roc_auc_mean':float(cv_final['test_roc_auc'].mean()),\n",
                "        'cv_roc_auc_std':float(cv_final['test_roc_auc'].std())\n",
                "    },\n",
                "    'targets':{'min_recall':0.80,'min_precision':0.65,'beta':1.5}\n",
                "}\n",
                "\n",
                "path = OUTPUT_DIR/'best_config.json'\n",
                "with open(path,'w') as f:\n",
                "    json.dump(config,f,indent=2)\n",
                "\n",
                "print('='*70)\n",
                "print('SAVED')\n",
                "print('='*70)\n",
                "print(f'✓ {path}')\n",
                "print(f'\\nModel: {best_model_name}')\n",
                "print(f'Features: {len(selected)}')\n",
                "print(f\"Recall: {config['performance']['cv_recall_mean']:.1%}\")\n",
                "print(f\"Precision: {config['performance']['cv_precision_mean']:.1%}\")\n",
                "print('\\n🚀 Ready for: cd ../3-cicd && python train.py')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write notebook
with open('experiment.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Notebook created: experiment.ipynb")
print("\n📂 Move it to the right folder:")
print("   mv experiment.ipynb '0 try/'")
print("\n🚀 Then run:")
print("   cd '0 try'")
print("   jupyter notebook experiment.ipynb")