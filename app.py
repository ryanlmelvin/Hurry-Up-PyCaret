import streamlit as st
import pandas as pd
from pycaret.classification import *

intro_text = """
On average, data scientists spend upwards of 50,000 hours setting up their models with PyCaret over the course of their careers.<br><br>
When I heard that, I thought to myself, "Why on earth don't these guys just hurry up?"<br><br>
So that's just what we did. Introducing the PyCaret Setup Accelerator, the fastest way to set up and compare classification models.<br><br>
With PyCaret, you can classify your data in a fraction of the time it takes to wait for your shrimp to arrive.<br><br>
So hurry up and try the PyCaret Setup Accelerator today, and get back to doing the things you love, like sipping margaritas on the beach or binge-watching your favorite TV show!
"""

st.markdown(intro_text, unsafe_allow_html=True)

st.title("PyCaret Classification App")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if 'df' in locals():
    target_column = st.selectbox("Select target column", df.columns)

    if target_column is not None:
        # Define PyCaret setup parameters in a dictionary
        setup_params = {
            "numeric_features": None,
            "categorical_features": None,
            "date_features": None,
            "ignore_features": None,
            "normalize": True,
            "normalize_method": "zscore",
            "transformation": False,
            "transformation_method": "yeo-johnson",
            "handle_unknown_categorical": True,
            "unknown_categorical_method": "least_frequent",
            "pca": False,
            "pca_method": "linear",
            "pca_components": None,
            "pca_iter": 100,
            "ignore_low_variance": False,
            "combine_rare_levels": False,
            "rare_level_threshold": 0.1,
            "bin_numeric_features": None,
            "remove_outliers": False,
            "outliers_threshold": 0.05,
            "remove_multicollinearity": False,
            "multicollinearity_threshold": 0.9,
            "create_clusters": False,
            "cluster_iter": 20,
            "polynomial_features": False,
            "polynomial_degree": 2,
            "trigonometry_features": False,
            "polynomial_threshold": 0.1,
            "group_features": None,
            "group_names": None,
            "feature_selection": False,
            "feature_selection_threshold": 0.8,
            "feature_interaction": False,
            "feature_ratio": False,
            "interaction_threshold": 0.01,
            "fix_imbalance": False,
            "fix_imbalance_method": None,
            "data_split_shuffle": True,
            "data_split_stratify": False,
            "fold_strategy": "kfold",
            "fold": 10,
            "fold_shuffle": False,
            "fold_groups": None,
            "n_jobs": -1,
            "use_gpu": False,
            "gpu_id": None,
            "silent": True,
        }

        with st.form(key="form"):
            # Show all available parameters in a form
            st.write("### PyCaret Setup Parameters")
            form_inputs = {}
            for param, value in setup_params.items():
                value = str(value) if value is not None else "None"
                form_inputs[param] = st.text_input(param, value, key=param)

            # Add a button to submit the form
            submit_button = st.form_submit_button(label="Compare Models")

            if submit_button:
                # Update the PyCaret setup configuration based on the form inputs
                for param, value in form_inputs.items():
                    if value != "None":
                        setup_params[param] = value

                # Set up the PyCaret classification task using the updated parameters
                clf_setup = setup(
                    data=df,
                    target=target_column,
                    **setup_params,
                )

                # Train and compare models using PyCaret
                best_model = compare_models()
                st.write("### Best Model")
                st.write(best_model)

