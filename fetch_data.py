import requests
import pandas as pd
from io import BytesIO
from Utils.config import DATASET_URLS

def fetch_real_data(domain):
    url = DATASET_URLS.get(domain)
    if not url:
        raise ValueError(f"No URL found for domain: {domain}")

    response = requests.get(url)
    response.raise_for_status()

    df = pd.read_csv(BytesIO(response.content))
    return df
