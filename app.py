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
            clf_setup_dict = eval(clf_setup)

            # Show all available parameters in a form
            st.write("### PyCaret Setup Parameters")
            form_inputs = {}
            params = get_config("prep_pipe")
            for param in params:
                default_value = clf_setup_dict["prep_pipe"][0][param]
                default_value = str(default_value) if default_value is not None else "None"
                text_input = st.text_input(param, default_value, key=param)
                form_inputs[param] = text_input

            # Add a button to submit the form
            submit_button = st.form_submit_button(label="Compare Models")

            if submit_button:
                # Update the PyCaret setup configuration based on the form inputs
                for param, text_input in form_inputs.items():
                    if text_input != "None":
                        clf_setup_dict["prep_pipe"][0][param] = text_input
                clf_setup_str = str(clf_setup_dict)

                # Train and compare models using PyCaret
                best_model = compare_models()
                st.write("### Best Model")
                st.write(best_model)
