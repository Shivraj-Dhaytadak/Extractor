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
                filtered_details = { k: v for k, v in details.items() if k not in ["rateCode", "appliesTo"] }
                combined_details = {
                    "productFamily": product_family,
                    "attributes": product_attributes,
                    **filtered_details
                }
                pricing_details.append(combined_details)
    
    return pricing_details

if __name__ == "__main__":
    json_file_path = os.path.join(os.path.dirname(__file__), 'json', 'AWSLambda.json')
    print(json_file_path)
    pricing_details = get_pricing_details(json_file_path)
    
    output_file = os.path.join(os.path.dirname(__file__), 'lambda.json')
    with open(output_file, 'w') as f:
        json.dump(pricing_details, f, indent=2)
    
    print(f"Pricing details successfully dumped to {output_file}")