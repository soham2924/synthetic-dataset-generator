import streamlit as st
import pandas as pd
import os

from generate_schema import generate_schema
from fetch_data import fetch_real_data
from synthetic_generator import train_and_generate_synthetic

# Set page config
st.set_page_config(page_title="AI Synthetic Dataset Generator", layout="wide")

st.title("✨ AI-Powered Synthetic Dataset Generator")
st.write("Give a short description of the dataset you need, and AI will generate it for you using real data + GANs!")

# Input fields
prompt = st.text_input("Describe the dataset (e.g., 'Create dataset for hospital patients'):")
domain = st.selectbox("Select Domain for Real Data", ["healthcare", "ecommerce"])

# Generate button
if st.button("Generate Dataset"):
    if not prompt.strip():
        st.error("Please enter a valid description.")
    else:
        with st.spinner("Generating schema using AI..."):
            schema = generate_schema(prompt)

        st.success("Schema Generated!")
        st.json(schema)

        with st.spinner(f"Fetching real {domain} data..."):
            real_data = fetch_real_data(domain)
            real_data = real_data[schema['columns']]

        st.success(f"Real data fetched — Shape: {real_data.shape}")

        # Train GAN and Generate Synthetic Data
        output_path = f"outputs/synthetic_{domain}.csv"
        with st.spinner("Training GAN and generating synthetic data..."):
            train_and_generate_synthetic(real_data, schema, output_path)

        st.success(f"Synthetic dataset saved as `{output_path}`")

        # Show preview of generated data
        synthetic_data = pd.read_csv(output_path)
        st.write("🔍 Preview of Synthetic Dataset:")
        st.dataframe(synthetic_data.head())

        # Download link
        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Dataset",
                data=f,
                file_name=f"synthetic_{domain}.csv",
                mime="text/csv"
            )
