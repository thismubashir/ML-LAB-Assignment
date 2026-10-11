# AIC354 Assignment 2 — Prediction Apps

**Name:** Mubashir Ali  
**Registration No.:** Fa24-bse-140

The package contains two project folders with simple Python code, executed Jupyter notebooks, datasets, trained models and Streamlit apps.

| Project | Package folder | GitHub notebook | Streamlit main file |
| --- | --- | --- | --- |
| Titanic | `titanic_deployment` | `Titanic_Passenger_Survival_Recommended.ipynb` | `app.py` |
| Heart Disease | `heart_disease_project` | `heart_disease_project/Heart_Disease_Prediction.ipynb` | `heart_disease_project/app.py` |

## Run locally

Open a terminal in the relevant project folder.

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook
```

To run its web app:

```bash
python -m streamlit run app.py
```

Run notebook cells from top to bottom. The notebook regenerates its trained model and CSV result files. The extra prediction `.py` file contains the same simple code.

## GitHub

[https://github.com/thismubashir/ML-LAB-Assignment](https://github.com/thismubashir/ML-LAB-Assignment)

Titanic files are at the repository root. Heart Disease files are in `heart_disease_project/`. Both apps use branch `main` and Python 3.12.

Titanic teaching source: https://github.com/haqnawaz99/ml-teaching-notebooks  
Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

## Streamlit Links

| App | Streamlit link |
| --- | --- |
| Titanic | [Open app](https://ml-lab-assignment-gwnyxabzax8nwfaueusdeb.streamlit.app/) |
| Heart Disease | [Open app](https://ml-lab-assignment-tbdhtsrxjgmpyfxc2vxktl.streamlit.app/) |
