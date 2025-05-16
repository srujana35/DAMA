import json

def filter_gender_examples(input_file, output_file):
    """
    Filter the test.json file to keep only examples where bias_type is 'gender'.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the filtered JSON file
    """
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Filter intrasentence examples
        filtered_intrasentence = [
            example for example in data['data']['intrasentence']
            if example['bias_type'].lower() == 'gender'
        ]
        
        # Create new data structure with only gender examples
        filtered_data = {
            'version': data['version'],
            'data': {
                'intrasentence': filtered_intrasentence,
                'intersentence': []  # Empty list since we're not using intersentence
            }
        }
        
        # Write the filtered data to output file
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"Original number of intrasentence examples: {len(data['data']['intrasentence'])}")
        print(f"Number of gender examples: {len(filtered_intrasentence)}")
        print(f"Filtered data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    input_file = "data/test.json"
    output_file = "data/test_gender.json"
    filter_gender_examples(input_file, output_file) 