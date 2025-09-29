import json
import os
from typing import List, Dict, Any

import json
import os
from typing import List, Dict, Any

def process_entry_to_cluster(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Combines all attacker transactions with the first profit detail transaction
    into a single, de-duplicated cluster.
    """
    cluster = []
    seen_hashes = set()

    # Get all attacker transactions
    attacker_txns = entry.get('attacker_transactions', [])
    for txn_dict in attacker_txns:
        if isinstance(txn_dict, dict):
            txn_hash = txn_dict.get('hash')
            if txn_hash and txn_hash not in seen_hashes:
                cluster.append(txn_dict)
                seen_hashes.add(txn_hash)

    # Get only the first element from profits_Details, if it exists
    profits_details = entry.get('profits_details', [])
    if isinstance(profits_details, list) and len(profits_details) > 0:
        first_profit_txn = profits_details[0]
        if isinstance(first_profit_txn, dict):
            txn_hash = first_profit_txn.get('hash')
            if txn_hash and txn_hash not in seen_hashes:
                cluster.append(first_profit_txn)
                seen_hashes.add(txn_hash)
    
    # You can also add other fields here like victim_details if needed
    victim_details = entry.get('victim_details')
    if isinstance(victim_details, dict):
        txn_hash = victim_details.get('hash')
        if txn_hash and txn_hash not in seen_hashes:
            cluster.append(victim_details)
            seen_hashes.add(txn_hash)

    return cluster


def main():
    """
    Main function to load the dataset, process each entry, and save the result.
    """
    input_file_path = "dataset_post_engineering.json"
    output_file_path = "clusterized_dataset.json"
    
    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: The input file was not found at {input_file_path}")
        return

    # Load the original data
    print(f"Loading data from {input_file_path}...")
    with open(input_file_path, 'r') as f:
        raw_data = json.load(f)

    processed_data = []
    
    # Process each entry in the dataset
    print("Processing entries...")
    for i, entry in enumerate(raw_data):
        # Create the new entry structure with a "cluster" of transactions
        new_entry = {
            "label": entry.get("label"),
            "cluster": process_entry_to_cluster(entry)
        }
        processed_data.append(new_entry)
        
    # Save the new dataset to a JSON file
    print(f"Saving processed data to {output_file_path}...")
    with open(output_file_path, 'w') as f:
        json.dump(processed_data, f, indent=2)

    print(f"Successfully processed {len(processed_data)} entries.")

if __name__ == "__main__":
    main()