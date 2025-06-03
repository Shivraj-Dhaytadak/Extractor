import json
import os

def get_pricing_details(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    products = data.get("products", {})
    pricing_details = []
    terms = data.get("terms", {}).get("OnDemand", {})

    for sku_key, sku_offers in terms.items():
        product_data = products.get(sku_key, {})
        product_family = product_data.get("productFamily", "N/A")
        product_attributes = product_data.get("attributes", {})

        for offer_key, offer_data in sku_offers.items():
            price_dimensions = offer_data.get("priceDimensions", {})
            for dim_key, details in price_dimensions.items():
                filtered_details = {k: v for k, v in details.items() if k not in ["rateCode", "appliesTo"]}
                combined_details = {
                    "productFamily": product_family,
                    "attributes": product_attributes,
                    **filtered_details
                }
                pricing_details.append(combined_details)

    return pricing_details

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".json"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"parsed_{file_name}")

            try:
                print(f"Processing: {input_path}")
                pricing_details = get_pricing_details(input_path)
                with open(output_path, 'w') as f:
                    json.dump(pricing_details, f, indent=2)
                print(f"✔ Successfully written to: {output_path}")
            except Exception as e:
                print(f"✘ Failed to process {file_name}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, 'json')
    output_dir = os.path.join(base_dir, 'parsed_output')
    
    process_directory(input_dir, output_dir)
