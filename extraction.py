import json
import os
import requests
def extract_services(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return list(data.get("offers", {}).keys())

def download_file(url, filename):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print()
        print(f"Failed to download {url} -> {e}")
        print()

def download_all_files(service_names, output_dir='C:\VScodeMaster\Inferencing\Extractor\download_json_csv'):
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws"

    for service in service_names:
        json_url = f"{base_url}/{service}/current/us-east-1/index.json"
        csv_url = f"{base_url}/{service}/current/us-east-1/index.csv"
        json_out_dir = os.path.join(output_dir, 'json')
        csv_out_dir = os.path.join(output_dir, 'csv')
        json_filename = os.path.join(json_out_dir, f"{service}.json")
        csv_filename = os.path.join(csv_out_dir, f"{service}.csv")

        download_file(json_url, json_filename)
        download_file(csv_url, csv_filename)

# Entry point
if __name__ == "__main__":
    input_json_file = 'C:\VScodeMaster\Inferencing\Extractor\index.json'  # Your local JSON with service list
    service_names = extract_services(input_json_file)
    download_all_files(service_names)
