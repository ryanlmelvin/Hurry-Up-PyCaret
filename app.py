import streamlit as st
import pandas as pd
from pycaret.classification import *

# Add a title and a file upload widget to your app:
st.title("Hurry Up Pycaret")

intro_text = """
On average, data scientists spend upwards of 50,000 hours setting up their models with PyCaret over the course of their careers.<br><br>
When I heard that, I thought to myself, "Why on earth don't these guys just hurry up?"<br><br>
So that's just what we did. Introducing the PyCaret Setup Accelerator, the fastest way to set up and compare classification models.<br><br>
With Hurry Up Pycaret, you can classify your data in a fraction of the time it takes to correctly setup a PyCaret experiment.<br><br>
So hurry up and try Hurry Up Pycaret today, and get back to doing the things you love, like sipping margaritas on the beach or binge-watching your favorite TV show!
"""

st.markdown(intro_text, unsafe_allow_html=True)

st.title("PyCaret Classification App")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

if 'df' in locals():
    target_column = st.selectbox("Select target column", df.columns)

    if target_column is not None:
        with st.form(key="form"):
            # Set up the PyCaret classification task
            clf_setup = setup(
                data=df,
                target=target_column,
                silent=True,  # Disable PyCaret output
            )

            # Show all available parameters in a form
            st.write("### PyCaret Setup Parameters")
            param_defaults = {
                "numeric_features": None,
                "categorical_features": None,
                "date_features": None,
                "ignore_features": None,
                "normalize": False,
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
                "fold_strategy": "stratifiedkfold",
                "fold": 10,
                "fold_shuffle": False,
                "fold_groups": None,
                "n_jobs": -1,
                "use_gpu": False,
                "gpu_id": 0,
                "silent": True,
            }
            form_inputs = {}
            for param, default_value in param_defaults.items():
                value = clf_setup[1].get(param, default_value)
                value = str(value) if value is not None else "None"
                new_value = st.text_input(param, value)
                form_inputs[param] = new_value

            # Add a button to submit the form
            submit_button = st.form_submit_button(label="Compare Models")

            if submit_button:
                # Update the PyCaret setup configuration based on the form inputs
                for param, value in form_inputs.items():
                    if value != "None":
                        clf_setup[1][param] = value

                # Train and compare models using PyCaret
                best_model = compare_models()
                st.write("### Best Model")
                st.write(best_model)
