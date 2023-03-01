import streamlit as st
import pandas as pd
from pycaret.classification import *

st.title("Hurry Up Pycaret")

intro_text = """
On average, data scientists spend upwards of 50,000 hours setting up their models with PyCaret over the course of their careers.<br><br>
When I heard that, I thought to myself, "Why on earth don't these guys just hurry up?"<br><br>
So that's just what we did. Introducing Hurry Up PyCaret, the fastest way to set up and compare classification models.<br><br>
With Hurry Up PyCaret, you can classify your data in a fraction of the time it takes to actually set up PyCaret.<br><br>
So hurry up and try Hurry Up PyCaret today, and get back to doing the things you love, like sipping margaritas on the beach or binge-watching your favorite TV show!
"""

st.markdown(intro_text, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a CSV")
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
            "normalize_method": ["zscore", "minmax", "maxabs", "robust"],
            "transformation": False,
            "transformation_method": ["yeo-johnson", "quantile", "yeo-johnson", None],
            "handle_unknown_categorical": True,
            "unknown_categorical_method": ["least_frequent", "most_frequent"],
            "pca": False,
            "pca_method": ["linear", "kernel"],
            "pca_components": None,
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
            "fold_strategy": ["kfold", "stratifiedkfold", "timeseries"],
            "fold": 10,
            "fold_shuffle": False,
            "fold_groups": None,
            "n_jobs": -1,
            "use_gpu": False,
            "silent": True,
        }

      with st.form(key="form"):
        # Show all available parameters in a form
        st.write("### PyCaret Setup Parameters")
        form_inputs = {}
        for param, value in setup_params.items():
            if isinstance(value, bool):
                # For boolean parameters, show a checkbox widget
                new_value = st.checkbox(param, value=value, key=param)
            elif isinstance(value, (int, float)):
                # For numeric parameters, show a number input widget
                new_value = st.number_input(param, value=value, key=param)
            elif isinstance(value, list):
                # For enumerable options, show a dropdown menu
                new_value = st.selectbox(param, value, key=param)
            else:
                # For all other parameters, show a text input widget
                new_value = st.text_input(param, str(value), key=param)
            form_inputs[param] = new_value

        # Add a button to submit the form
        submit_button = st.form_submit_button(label="Compare Models")

        if submit_button:
            # Update the PyCaret setup configuration based on the form inputs
            for param, value in setup_params.items():
                if isinstance(setup_params[param], bool):
                    # For boolean parameters, convert the string to a boolean value
                    form_inputs[param] = str(form_inputs[param]).lower() == "true"
                elif isinstance(setup_params[param], float):
                    # For numeric parameters, convert the string to a float value
                    form_inputs[param] = float(form_inputs[param])
                elif isinstance(setup_params[param], int):
                    # For numeric parameters, convert the string to an integer value
                    form_inputs[param] = int(form_inputs[param])
                else:
                    # For all other parameters, use the string value
                    form_inputs[param] = form_inputs[param]

            # Set up the PyCaret classification task using the updated parameters
            clf_setup = setup(
                data=df,
                target=target_column,
                **form_inputs,
            )

            # Train and compare models using PyCaret
            best_model = compare_models()
            results = pull()
            st.write("### Best Model")
            st.write(results)
