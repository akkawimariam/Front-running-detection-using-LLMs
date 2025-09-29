import json
from collections import defaultdict

# === CONFIGURATION ===
INPUT_FILE = "fully_labeled_dataset.json"  # Original input file
INTERMEDIATE_FILE = "single_label_dataset.json"  # File after label conversion
OUTPUT_FILE = "dataset_post_engineering.json"  # Final cleaned output file

FIELDS_TO_REMOVE = [
    "bytecode_similarity", "addresses_distinct", "attacker_gas_higher",
    "flow_consistency", "token_amount_match", "token_contract_match",
    "distinct_hashes", "tx_order_correct", "gas_price_order_correct",
    "cluster_size", "min_gas_used", "min_gas_util_ratio", "ClusterSize>1",
    "AllGas>21k", "GasUtil>99%", "token_identifier_match", "matched_token_type",
    "displacement_label", "insertion_label", "suppression_label"  # Added the old label fields
]

def convert_to_single_label(input_file, output_file):
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Track statistics
    stats = {
        'total_entries': len(data),
        'displacement_count': 0,
        'insertion_count': 0,
        'suppression_count': 0,
        'conflicting_entries': 0,
        'conflicting_details': [],
        'no_label_entries': 0,
        'no_label_details': []
    }
    
    converted_data = []
    
    for idx, entry in enumerate(data):
        displacement = entry.get('displacement_label', 0)
        insertion = entry.get('insertion_label', 0)
        suppression = entry.get('suppression_label', 0)
        
        # Count how many labels are set to 1
        label_count = displacement + insertion + suppression
        
        if label_count == 0:
            # No label found
            stats['no_label_entries'] += 1
            stats['no_label_details'].append({
                'index': idx,
                'entry_id': entry.get('id', f'entry_{idx}'),
                'displacement': displacement,
                'insertion': insertion,
                'suppression': suppression
            })
            # Assign -1 (unknown)
            new_entry = entry.copy()
            new_entry['label'] = -1
            converted_data.append(new_entry)
            
        elif label_count > 1:
            # Multiple labels found - CONFLICT!
            stats['conflicting_entries'] += 1
            stats['conflicting_details'].append({
                'index': idx,
                'entry_id': entry.get('id', f'entry_{idx}'),
                'displacement': displacement,
                'insertion': insertion,
                'suppression': suppression,
                'labels_found': []
            })
            
            if displacement == 1:
                stats['conflicting_details'][-1]['labels_found'].append('displacement')
            if insertion == 1:
                stats['conflicting_details'][-1]['labels_found'].append('insertion')
            if suppression == 1:
                stats['conflicting_details'][-1]['labels_found'].append('suppression')
            
            # Priority: displacement > insertion > suppression
            
            if insertion == 1:
                new_label = 1
                stats['insertion_count'] += 1
            elif displacement == 1:
                new_label = 0
                stats['displacement_count'] += 1
            elif suppression == 1:
                new_label = 2
                stats['suppression_count'] += 1
                
            new_entry = entry.copy()
            new_entry['label'] = new_label
            converted_data.append(new_entry)
            
        else:
            # Single label found - perfect case
            new_entry = entry.copy()
            if displacement == 1:
                new_entry['label'] = 0
                stats['displacement_count'] += 1
            elif insertion == 1:
                new_entry['label'] = 1
                stats['insertion_count'] += 1
            elif suppression == 1:
                new_entry['label'] = 2
                stats['suppression_count'] += 1
                
            converted_data.append(new_entry)
    
    # Remove the old label fields
    for entry in converted_data:
        entry.pop('displacement_label', None)
        entry.pop('insertion_label', None)
        entry.pop('suppression_label', None)
    
    # Save the converted dataset
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    # Print comprehensive report
    print("=" * 60)
    print("DATABASE CONVERSION REPORT")
    print("=" * 60)
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Displacement entries:    {stats['displacement_count']}")
    print(f"Insertion entries:       {stats['insertion_count']}")
    print(f"Suppression entries:     {stats['suppression_count']}")
    print(f"No label entries:        {stats['no_label_entries']}")
    print(f"Conflicting entries:     {stats['conflicting_entries']}")
    print("=" * 60)
    
    if stats['conflicting_entries'] > 0:
        print("\nCONFLICTING ENTRIES FOUND:")
        print("=" * 40)
        for i, conflict in enumerate(stats['conflicting_details'][:10]):  # Show first 10
            print(f"{i+1}. Entry {conflict['entry_id']} (index {conflict['index']})")
            print(f"   Labels found: {', '.join(conflict['labels_found'])}")
            print(f"   Values: displacement={conflict['displacement']}, "
                  f"insertion={conflict['insertion']}, "
                  f"suppression={conflict['suppression']}")
        if len(stats['conflicting_details']) > 10:
            print(f"   ... and {len(stats['conflicting_details']) - 10} more conflicts")
    
    if stats['no_label_entries'] > 0:
        print(f"\nNO LABEL ENTRIES: {stats['no_label_entries']}")
        print("=" * 30)
        for i, no_label in enumerate(stats['no_label_details'][:5]):  # Show first 5
            print(f"{i+1}. Entry {no_label['entry_id']} (index {no_label['index']})")
            print(f"   Values: displacement={no_label['displacement']}, "
                  f"insertion={no_label['insertion']}, "
                  f"suppression={no_label['suppression']}")
        if len(stats['no_label_details']) > 5:
            print(f"   ... and {len(stats['no_label_details']) - 5} more unlabeled entries")
    
    # Save detailed report to file
    report_file = "conversion_report.json"
    with open(report_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"Converted dataset saved to: {output_file}")
    print(f"Detailed report saved to: {report_file}")
    
    return stats, converted_data

def clean_json(data, output_file, fields_to_remove):
    # Process each entry
    for entry in data:
        for field in fields_to_remove:
            entry.pop(field, None)  # safely remove if exists

    # Save the cleaned dataset
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Cleaned dataset saved to {output_file}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Step 1: Convert the multi-label format to single label format
    print("Starting label conversion process...")
    stats, converted_data = convert_to_single_label(INPUT_FILE, INTERMEDIATE_FILE)
    
    # Step 2: Clean the converted data by removing specified fields
    print("\nStarting field removal process...")
    clean_json(converted_data, OUTPUT_FILE, FIELDS_TO_REMOVE)
    
    print("\nProcessing complete! Final output saved to:", OUTPUT_FILE)