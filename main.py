import argparse
import pandas as pd
from generate_schema import generate_schema
from fetch_data import fetch_real_data
from synthetic_generator import train_and_generate_synthetic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Describe the dataset you want")
    parser.add_argument("--domain", type=str, default="healthcare", help="Domain to fetch real data from (optional)")
    args = parser.parse_args()

    # Step 1: Generate schema using LLM
    schema = generate_schema(args.prompt)
    print(f"📊 Generated schema: {schema}")

    # Step 2: Fetch real data (optional)
    real_data = fetch_real_data(args.domain)

    # Step 3: Preprocess (if necessary)
    real_data = real_data[schema['columns']]  # Match columns from schema
    print(f"✅ Fetched real data with shape: {real_data.shape}")

    # Step 4: Train GAN and generate synthetic data
    output_path = f"outputs/synthetic_{args.domain}.csv"
    train_and_generate_synthetic(real_data, schema, output_path)

if __name__ == "__main__":
    main()
