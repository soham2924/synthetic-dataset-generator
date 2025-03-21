import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder
import os

def train_and_generate_synthetic(real_data, schema, output_path):
    categorical_cols = [col for col, dtype in zip(schema['columns'], schema['types']) if dtype == 'string']

    # Encode categorical columns for GAN training
    for col in categorical_cols:
        le = LabelEncoder()
        real_data[col] = le.fit_transform(real_data[col])

    # Train CTGAN
    gan = CTGAN(epochs=300)
    gan.fit(real_data, categorical_cols)

    # Generate synthetic data
    synthetic_data = gan.sample(schema['size'])

    # Decode categorical columns
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(real_data[col])
        synthetic_data[col] = le.inverse_transform(synthetic_data[col])

    # Save to CSV
    os.makedirs('outputs', exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    print(f"✅ Synthetic data saved to {output_path}")
