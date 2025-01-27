# import pickle
# import json
# import os
# import numpy as np

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         return super().default(obj)

# def convert_pkl_to_json(pkl_path):
#     # Extract the directory and filename from the path
#     directory = os.path.dirname(pkl_path)
#     filename = os.path.basename(pkl_path)
    
#     # Create the new json filename by replacing .pkl extension
#     json_filename = filename.replace('.pkl', '.json')
#     json_path = os.path.join(directory, json_filename)
    
#     # Read the pickle file
#     with open(pkl_path, 'rb') as pkl_file:
#         data = pickle.load(pkl_file)
    
#     # Write to JSON file using the custom encoder
#     with open(json_path, 'w') as json_file:
#         json.dump(data, json_file, indent=4, cls=NumpyEncoder)
    
#     print(f"Successfully converted {pkl_path} to {json_path}")

# def convert_folder_pkl_to_json(folder_path):
#     # Get all .pkl files in the folder
#     pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    
#     if not pkl_files:
#         print(f"No .pkl files found in {folder_path}")
#         return
    
#     # Process each .pkl file
#     for pkl_file in pkl_files:
#         pkl_path = os.path.join(folder_path, pkl_file)
#         try:
#             convert_pkl_to_json(pkl_path)
#         except Exception as e:
#             print(f"Error converting {pkl_path}: {str(e)}")

# if __name__ == "__main__":
#     folder_path = "/home/sina/projects/VLM-Uncertainty-Bench/output_llm/qwen-7b"
#     convert_folder_pkl_to_json(folder_path)





# import json
# import os

# def merge_json_files(reference_path, logits_path, output_dir):
#     # Read the reference (main) JSON file
#     with open(reference_path, 'r') as f:
#         reference_data = json.load(f)
    
#     # Read the logits JSON file
#     with open(logits_path, 'r') as f:
#         logits_data = json.load(f)
    
#     # Get just the filename for output
#     output_filename = os.path.basename(reference_path)
    
#     # Create a dictionary for faster lookup of logits by ID
#     logits_map = {item['id']: item['logits_options'] for item in logits_data}
    
#     # Update reference data with logits using the new key name
#     for item in reference_data:
#         if item['id'] in logits_map:
#             item['logits'] = logits_map[item['id']]
    
#     # Write the merged data to a new JSON file
#     output_path = os.path.join(output_dir, output_filename)
#     with open(output_path, 'w') as f:
#         json.dump(reference_data, f, indent=4)
    
#     print(f"Successfully merged and saved: {output_path}")

# def process_all_files():
#     # Define paths
#     reference_dir = '/home/sina/projects/VLM-Uncertainty-Bench/datasets_llm'
#     logits_dir = '/home/sina/projects/VLM-Uncertainty-Bench/output_llm/qwen-7b'
#     output_dir = '/home/sina/projects/VLM-Uncertainty-Bench/output_llm/qwen-7b'
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Get all JSON files in reference directory
#     reference_files = [f for f in os.listdir(reference_dir) if f.endswith('.json')]
    
#     # Process each reference file
#     for ref_file in reference_files:
#         ref_path = os.path.join(reference_dir, ref_file)
        
#         # Look for matching file in logits directory
#         # Remove .json extension for matching
#         ref_name = ref_file[:-5]  # remove '.json'
        
#         # Find matching logits file
#         matching_logits = None
#         for logits_file in os.listdir(logits_dir):
#             if logits_file.endswith('.json') and ref_name in logits_file:
#                 matching_logits = os.path.join(logits_dir, logits_file)
#                 break
        
#         if matching_logits:
#             try:
#                 merge_json_files(ref_path, matching_logits, output_dir)
#                 print(f"Processed {ref_file} with {os.path.basename(matching_logits)}")
#             except Exception as e:
#                 print(f"Error processing {ref_file}: {str(e)}")
#         else:
#             print(f"No matching logits file found for {ref_file}")

# if __name__ == "__main__":
#     process_all_files()




import pickle
import json
import os
import numpy as np

def convert_json_to_pkl(json_path):
    # Extract the directory and filename from the path
    directory = os.path.dirname(json_path)
    filename = os.path.basename(json_path)
    
    # Create the new pickle filename by replacing .json extension
    pkl_filename = filename.replace('.json', '.pkl')
    pkl_path = os.path.join(directory, pkl_filename)
    
    # Read the JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Convert data to ensure logits are numpy arrays
    for item in data:
        if 'logits' in item:
            # Convert logits to numpy float32 array
            item['logits'] = np.array(item['logits'], dtype=np.float32)
    
    # Write to pickle file
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)
    
    print(f"Successfully converted {json_path} to {pkl_path}")
    # Verify the first item's logits type
    print(f"Sample logits type: {type(data[0]['logits'])}")
    print(f"Sample logits dtype: {data[0]['logits'].dtype}")

def process_folder(folder_path):
    # Get all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return
    
    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        try:
            convert_json_to_pkl(json_path)
        except Exception as e:
            print(f"Error converting {json_file}: {str(e)}")

if __name__ == "__main__":
    # Example usage
    folder_path = "/home/sina/projects/VLM-Uncertainty-Bench/output_llm/qwen-7b"
    process_folder(folder_path)



# import json
# import os

# def rename_json_key(json_path):
#     # Read the JSON file
#     with open(json_path, 'r') as file:
#         data = json.load(file)
    
#     # Ensure data is a list/array
#     if not isinstance(data, list):
#         raise ValueError("JSON file must contain an array of objects")
    
#     # Rename the key for each object
#     for item in data:
#         if 'logits_options' in item:
#             item['logits'] = item.pop('logits_options')
    
#     # Write back to the same file
#     with open(json_path, 'w') as file:
#         json.dump(data, file, indent=4)
    
#     print(f"Successfully updated {json_path}")

# def process_folder(folder_path):
#     # Get all JSON files in the folder
#     json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
#     if not json_files:
#         print(f"No JSON files found in {folder_path}")
#         return
    
#     # Process each JSON file
#     for json_file in json_files:
#         json_path = os.path.join(folder_path, json_file)
#         try:
#             rename_json_key(json_path)
#         except Exception as e:
#             print(f"Error processing {json_file}: {str(e)}")

# if __name__ == "__main__":
#     # Example usage
#     folder_path = "/home/sina/projects/VLM-Uncertainty-Bench/output_llm/qwen-14b"
#     process_folder(folder_path)