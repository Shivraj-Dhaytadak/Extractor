import json

def extract_services(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract service names from the "offers" key
    services = list(data.get("offers", {}).keys())
    return services

def write_services_to_file(services, output_file):
    with open(output_file, 'w') as file:
        for service in services:
            file.write(f"{service}\n")

# Example usage
if __name__ == "__main__":
    input_json_file = 'index.json'    # Replace with your JSON file path
    output_txt_file = 'aws_services_list.txt'  # Output file path

    service_names = extract_services(input_json_file)
    write_services_to_file(service_names, output_txt_file)

    print(f"Service names written to {output_txt_file}")
