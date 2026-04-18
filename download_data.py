import os
import requests

def download_pdf(url, output_path):
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded successfully to {output_path}")

if __name__ == "__main__":
    url = "https://arxiv.org/pdf/2407.01449.pdf"
    output_path = "data/colpali_paper.pdf"
    download_pdf(url, output_path)
