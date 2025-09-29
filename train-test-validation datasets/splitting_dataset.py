import json
import random
import os

def split_datasets_simple(input_file, train_output, val_output, test_output, train_size=0.8, val_size=0.1, test_size=0.1):
    """
    Simple split: shuffle each class, then take first 80% for training, next 10% for validation, last 10% for testing.
    This guarantees no duplicates between splits.
    """
    # Validate the size parameters
    if abs(train_size + val_size + test_size - 1.0) > 0.001:
        raise ValueError(f"train_size + val_size + test_size must equal 1.0, got {train_size + val_size + test_size}")
    
    # Load the dataset
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} total entries from balanced dataset")
    
    # Split by labels
    label0 = [entry for entry in data if entry.get("label") == 0]  # displacement
    label1 = [entry for entry in data if entry.get("label") == 1]  # insertion
    label2 = [entry for entry in data if entry.get("label") == 2]  # suppression
    
    print(f"Original distribution: 0={len(label0)}, 1={len(label1)}, 2={len(label2)}")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle each class
    random.shuffle(label0)
    random.shuffle(label1) 
    random.shuffle(label2)
    
    # Split each class into train/val/test
    def split_class(entries, class_name):
        total = len(entries)
        train_count = int(total * train_size)
        val_count = int(total * val_size)
        
        train_entries = entries[:train_count]
        val_entries = entries[train_count:train_count + val_count]
        test_entries = entries[train_count + val_count:]
        
        print(f"{class_name}: {total} total -> Train: {len(train_entries)}, Val: {len(val_entries)}, Test: {len(test_entries)}")
        return train_entries, val_entries, test_entries
    
    # Split each class
    train0, val0, test0 = split_class(label0, "Displacement")
    train1, val1, test1 = split_class(label1, "Insertion")
    train2, val2, test2 = split_class(label2, "Suppression")
    
    # Combine all splits
    train_set = train0 + train1 + train2
    val_set = val0 + val1 + val2
    test_set = test0 + test1 + test2
    
    # Shuffle the final sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)
    
    # Create output directory
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    
    # Save datasets
    with open(train_output, 'w') as f:
        json.dump(train_set, f, indent=2)
    with open(val_output, 'w') as f:
        json.dump(val_set, f, indent=2)
    with open(test_output, 'w') as f:
        json.dump(test_set, f, indent=2)
    
    # Verify no duplicates by checking counts
    total_original = len(label0) + len(label1) + len(label2)
    total_split = len(train_set) + len(val_set) + len(test_set)
    
    print(f"\n✅ Split completed successfully!")
    print(f"Original total: {total_original}")
    print(f"Split total: {total_split}")
    print(f"Difference: {total_original - total_split} (should be 0)")
    
    # Final distribution
    def count_labels(entries, set_name):
        counts = {0: 0, 1: 0, 2: 0}
        for entry in entries:
            label = entry.get('label')
            if label in counts:
                counts[label] += 1
        print(f"{set_name}: 0={counts[0]}, 1={counts[1]}, 2={counts[2]}")
        return counts
    
    print(f"\nFinal distribution:")
    train_counts = count_labels(train_set, "Train")
    val_counts = count_labels(val_set, "Val")
    test_counts = count_labels(test_set, "Test")
    
    # Verify proportions
    print(f"\nProportions (should be close to {train_size*100:.0f}%/{val_size*100:.0f}%/{test_size*100:.0f}%):")
    for label in [0, 1, 2]:
        total = train_counts[label] + val_counts[label] + test_counts[label]
        if total > 0:
            train_pct = (train_counts[label] / total) * 100
            val_pct = (val_counts[label] / total) * 100
            test_pct = (test_counts[label] / total) * 100
            print(f"Label {label}: Train={train_pct:.1f}%, Val={val_pct:.1f}%, Test={test_pct:.1f}%")
    
    print(f"\nSaved to:")
    print(f"  - Training: {train_output}")
    print(f"  - Validation: {val_output}")
    print(f"  - Testing: {test_output}")

# Usage
if __name__ == "__main__":
    split_datasets_simple(
        input_file="balanced_dataset.json",
        train_output="dataset/train.json",
        val_output="dataset/validation.json", 
        test_output="dataset/test.json",
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )