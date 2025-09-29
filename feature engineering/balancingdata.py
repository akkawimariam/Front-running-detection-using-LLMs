import json
import random

def load_and_sample(json_path, out_path, n0=1500, n1=1500):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split by labels
    label0 = [entry for entry in data if entry.get("label") == 0]  # displacement
    label1 = [entry for entry in data if entry.get("label") == 1]  # insertion  
    label2 = [entry for entry in data if entry.get("label") == 2]  # suppression

    print(f"Original counts: 0={len(label0)}, 1={len(label1)}, 2={len(label2)}")

    # Simple unique identifier
    def get_entry_id(entry):
        """Get unique ID for an entry"""
        # Use main hash if available
        if entry.get('hash'):
            return entry['hash']
        
        # Fallback to hashing the entire entry
        return str(hash(json.dumps(entry, sort_keys=True)))

    # Remove duplicates within displacement class
    print("\nProcessing displacement class (0)...")
    displacement_unique = []
    displacement_seen = set()
    
    for entry in label0:
        entry_id = get_entry_id(entry)
        if entry_id not in displacement_seen:
            displacement_seen.add(entry_id)
            displacement_unique.append(entry)
    
    print(f"Displacement: {len(label0)} → {len(displacement_unique)} unique entries")

    # Remove duplicates within insertion class  
    print("Processing insertion class (1)...")
    insertion_unique = []
    insertion_seen = set()
    
    for entry in label1:
        entry_id = get_entry_id(entry)
        if entry_id not in insertion_seen:
            insertion_seen.add(entry_id)
            insertion_unique.append(entry)
    
    print(f"Insertion: {len(label1)} → {len(insertion_unique)} unique entries")

    # Keep all suppression entries (no sampling needed)
    print("Processing suppression class (2)...")
    suppression_all = label2  # Keep all suppression entries
    
    # Sample 1600 from each class (or all if less than 1600)
    sample_displacement = random.sample(displacement_unique, min(n0, len(displacement_unique)))
    sample_insertion = random.sample(insertion_unique, min(n1, len(insertion_unique)))
    
    # Final dataset: sampled displacement + sampled insertion + all suppression
    final_data = sample_displacement + sample_insertion + suppression_all
    
    # Final check for cross-class duplicates
    final_unique = []
    final_seen = set()
    cross_duplicates = 0
    
    for entry in final_data:
        entry_id = get_entry_id(entry)
        if entry_id not in final_seen:
            final_seen.add(entry_id)
            final_unique.append(entry)
        else:
            cross_duplicates += 1
            print(f"Cross-class duplicate found and removed: {entry_id}")

    random.shuffle(final_unique)

    # Save the balanced dataset
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_unique, f, indent=2)

    # Calculate final distribution
    label_counts = {0: 0, 1: 0, 2: 0}
    for entry in final_unique:
        label = entry.get('label')
        if label in label_counts:
            label_counts[label] += 1

    print(f"\n✅ Final balanced dataset: {len(final_unique)} UNIQUE entries")
    print(f"Class distribution:")
    print(f"  Displacement (0): {label_counts[0]} entries")
    print(f"  Insertion (1): {label_counts[1]} entries")  
    print(f"  Suppression (2): {label_counts[2]} entries")
    print(f"Cross-class duplicates removed: {cross_duplicates}")

# Usage
load_and_sample("clusterized_dataset.json", "balanced_dataset.json")