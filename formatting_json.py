import json

# Path to the original JSON file
input_file = '/home/sina/projects/VLM-Uncertainty-Bench/datasets/seedbench/seedbench.json'
# Path to save the formatted JSON file
output_file = '/home/sina/projects/VLM-Uncertainty-Bench/datasets/seedbench/formatted_file.json'

# Read and format the JSON
with open(input_file, 'r') as file:
    data = json.load(file)

# Write the formatted JSON with indentation
with open(output_file, 'w') as file:
    json.dump(data, file, indent=4)

print(f"Formatted JSON saved to {output_file}")
