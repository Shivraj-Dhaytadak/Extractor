import json

def extract_aws_pricing(json_data):
    products = json_data.get("products", {})
    terms = json_data.get("terms", {}).get("OnDemand", {})

    pricing_info = []

    for sku, product_data in products.items():
        attributes = product_data.get("attributes", {})
        product_family = product_data.get("productFamily", "")
        service_name = attributes.get("servicename", "")
        region = attributes.get("location", "")
        usage_type = attributes.get("usagetype", "")
        storage_type = attributes.get("storageType", "")
        
        # Match terms
        sku_terms = terms.get(sku, {})
        for term_key, term_data in sku_terms.items():
            for rate_code, price_data in term_data.get("priceDimensions", {}).items():
                price_per_unit = price_data.get("pricePerUnit", {}).get("USD", "N/A")
                unit = price_data.get("unit", "")
                description = price_data.get("description", "")
                pricing_info.append({
                    "service_name": service_name,
                    "product_family": product_family,
                    "region": region,
                    "usage_type": usage_type,
                    "storage_type": storage_type,
                    "price_per_unit": float(price_per_unit),
                    "unit": unit,
                    "description": description
                })
    
    return pricing_info
