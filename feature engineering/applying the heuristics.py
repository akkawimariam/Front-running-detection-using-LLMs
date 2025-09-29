import json
import os
from collections import Counter, defaultdict

# === CONFIGURATION ===
INPUT_FILE = "dataset_with_all_info_needed.json"  # Starting input file
DIS_OUTPUT_FILE = "dataset_with_dis_heuristics.json"    # After displacement labeling
INS_OUTPUT_FILE = "dataset_with_ins_heuristics.json"    # After insertion labeling  
SUP_OUTPUT_FILE = "dataset_with_sup_heuristics.json"    # After suppression labeling
FINAL_OUTPUT_FILE = "fully_labeled_dataset.json"        # Final combined output

def hex_to_int(value):
    """Convert hex string to integer, handling various formats"""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("0x"):
        return int(value, 16)
    try:
        return int(value)
    except:
        return 0

def get_victim_transaction_index(victim_details):
    """Safely extract transaction index from victim_details, handling nested lists"""
    # If it's a dictionary, return transactionIndex directly
    if isinstance(victim_details, dict):
        return victim_details.get("transactionIndex", -1)
    
    # If it's a list, recursively search for dictionaries
    if isinstance(victim_details, list):
        for item in victim_details:
            # Recursively search nested lists
            if isinstance(item, (dict, list)):
                result = get_victim_transaction_index(item)
                if result != -1:
                    return result
    return -1

def calculate_bytecode_similarity(input_a, input_v):
    """Calculate similarity ratio per paper specification"""
    # Remove '0x' prefix if present
    hex_a = input_a[2:] if input_a.startswith('0x') else input_a
    hex_v = input_v[2:] if input_v.startswith('0x') else input_v
    
    # Split into 4-byte chunks (8 hex characters)
    chunks_ta = [hex_a[i:i+8] for i in range(0, len(hex_a), 8) if i+8 <= len(hex_a)]
    chunks_tv = [hex_v[i:i+8] for i in range(0, len(hex_v), 8) if i+8 <= len(hex_v)]
    
    # Handle empty inputs
    if not chunks_ta or not chunks_tv:
        return 0.0
    
    # Create set for fast lookup
    set_ta = set(chunks_ta)
    set_tv = set(chunks_tv)
    
    # Count matching chunks (bidirectional)
    matches = len(set_ta & set_tv)
    
    # Ratio = matches / min(len(ta), len(tv)) as per paper
    min_length = min(len(chunks_ta), len(chunks_tv))
    return matches / min_length

def is_displacement(entry):
    """Determine if an entry is a displacement attack using heuristics"""
    v = entry.get("victim_details", [{}])
    a = entry.get("attacker_details", [{}])

    # Handle if they are lists
    if isinstance(v, list) and v:
        v = v[0]
    if isinstance(a, list) and a:
        a = a[0]

    # Default values
    cond1 = cond2 = cond3 = False
    similarity_ratio = 0.0

    # Heuristic 1: All addresses must be distinct
    try:
        cond1 = (a.get("sender") != v.get("sender") and a.get("receiver") != v.get("receiver"))
    except Exception:
        pass

    # Heuristic 2: Attacker gas price > Victim gas price
    try:
        cond2 = int(a.get("gasPrice", 0)) > int(v.get("gasPrice", 0))
    except Exception:
        pass

    # Heuristic 3: Input bytecode similarity >= 25%
    try:
        input_a = a.get("input")
        input_v = v.get("input")
        if input_a and input_v:
            similarity_ratio = calculate_bytecode_similarity(input_a, input_v)
            cond3 = (similarity_ratio >= 0.25)
    except Exception:
        pass

    displacement_label = int(cond1 and cond2 and cond3)
    return displacement_label, cond1, cond2, similarity_ratio

def find_attack_transfers(ea1, ev, ea2_list, attacker_addr):
    """Find coherent set of token transfers across all 3 transactions for all token types"""
    
    # Get ALL token transfers regardless of type
    ea1_transfers = ea1.get("token_transfers", [])
    ev_transfers = ev.get("token_transfers", [])
    
    valid_combinations = []
    
    for t1 in ea1_transfers:
        # Only consider transfers TO the attacker in EA1
        if t1.get("to", "").lower() != attacker_addr:
            continue
            
        exchange_addr = t1.get("from", "").lower()
        token_contract = t1.get("rawContract", {}).get("address", "").lower()
        token_type = t1.get("tokenType")
        
        # Find ALL matching EV transfers (could be multiple)
        matching_ev_transfers = []
        for t2 in ev_transfers:
            if (t2.get("to", "").lower() == exchange_addr and
                t2.get("rawContract", {}).get("address", "").lower() == token_contract and
                t2.get("tokenType") == token_type):
                matching_ev_transfers.append(t2)
        
        if not matching_ev_transfers:
            continue
            
        # For each matching EV transfer, find ALL matching EA2 transfers
        for t2 in matching_ev_transfers:
            for ea2 in ea2_list:
                ea2_transfers = ea2.get("token_transfers", [])
                
                matching_ea2_transfers = []
                for t3 in ea2_transfers:
                    if (t3.get("from", "").lower() == attacker_addr and
                        t3.get("to", "").lower() == exchange_addr and
                        t3.get("rawContract", {}).get("address", "").lower() == token_contract and
                        t3.get("tokenType") == token_type):
                        
                        # Check token-specific matching
                        if token_type == "ERC20":
                            token_value = hex_to_int(t1.get("value", 0))
                            t3_value = hex_to_int(t3.get("value", 0))
                            if token_value > 0 and t3_value > 0:
                                max_amount = max(token_value, t3_value)
                                if abs(t3_value - token_value) / max_amount <= 0.01:
                                    matching_ea2_transfers.append(t3)
                                
                        elif token_type == "ERC721":
                            token_id_t1 = t1.get("tokenId")
                            token_id_t3 = t3.get("tokenId")
                            if token_id_t1 is not None and token_id_t3 is not None and token_id_t1 == token_id_t3:
                                matching_ea2_transfers.append(t3)
                                
                        elif token_type == "ERC1155":
                            is_batch_t1 = t1.get("is_batch", False)
                            is_batch_t3 = t3.get("is_batch", False)
                            
                            if is_batch_t1 and is_batch_t3:
                                # Batch transfer: match token_ids arrays
                                token_ids_t1 = set(t1.get("token_ids", []))
                                token_ids_t3 = set(t3.get("token_ids", []))
                                if token_ids_t1 and token_ids_t3 and token_ids_t1 == token_ids_t3:
                                    matching_ea2_transfers.append(t3)
                            elif not is_batch_t1 and not is_batch_t3:
                                # Single transfer: match tokenId
                                token_id_t1 = t1.get("tokenId")
                                token_id_t3 = t3.get("tokenId")
                                if token_id_t1 is not None and token_id_t3 is not None and token_id_t1 == token_id_t3:
                                    matching_ea2_transfers.append(t3)
                
                # Store all valid combinations for this t1, t2 pair
                for t3 in matching_ea2_transfers:
                    valid_combinations.append((t1, t2, t3, ea2))
    
    # Return the best matching combination (prioritize ERC20, then by value/rarity)
    if valid_combinations:
        # Sort by token type priority: ERC20 > ERC1155 > ERC721
        def sort_key(combination):
            t1, t2, t3, ea2 = combination
            token_type = t1.get("tokenType")
            if token_type == "ERC20":
                return 0
            elif token_type == "ERC1155":
                return 1
            else:  # ERC721
                return 2
        
        valid_combinations.sort(key=sort_key)
        return valid_combinations[0]  # Return the highest priority combination
    
    return None, None, None, None

def is_insertion(entry, condition_stats, failure_reasons):
    try:
        # Initial validation
        ea1 = entry.get("attacker_details")
        ev = entry.get("victim_details")
        ea2_list = entry.get("profits_details", [])
        
        if not all([ea1, ev, ea2_list]):
            failure_reasons['missing_transaction_details'] += 1
            return 0, 0

        # Block number validation
        attacker_addr = ea1.get("receiver", "").lower()
        block_numbers = {
            hex_to_int(ea1.get("blockNumber")),
            hex_to_int(ev.get("blockNumber")),
        }
        for profit in ea2_list:
            block_numbers.add(hex_to_int(profit.get("blockNumber")))
            
        if len(block_numbers) != 1:
            failure_reasons['multiple_blocks'] += 1
            return 0, 0

        # Find matching transfers
        t1, t2, t3, ea2_candidate = find_attack_transfers(ea1, ev, ea2_list, attacker_addr)
        if not all([t1, t2, t3]):
            failure_reasons['no_matching_transfers'] += 1
            return 0, 0

        # Heuristic checks
        passed = {f"cond{i}": False for i in range(1, 7)}
        token_type = t1.get("tokenType")
        
        # CORRECTED CONDITION 1: Flow consistency
        sA1 = t1.get("from", "").lower()        # Exchange address
        rA1 = t1.get("to", "").lower()          # Attacker address
        rV = t2.get("to", "").lower()           # Exchange address (receiver in EV)
        sA2 = t3.get("from", "").lower()        # Attacker address
        rA2 = t3.get("to", "").lower()          # Exchange address
        
        passed["cond1"] = (sA1 == rV == rA2) and (rA1 == sA2)

        # Condition 2: Token identifier match (type-specific)
        if token_type == "ERC20":
            aA1 = hex_to_int(t1.get("value", 0))
            aA2 = hex_to_int(t3.get("value", 0))
            if aA1 > 0 and aA2 > 0:
                max_amount = max(aA1, aA2)
                passed["cond2"] = (abs(aA1 - aA2) / max_amount <= 0.01)
            else:
                passed["cond2"] = False
                
        elif token_type == "ERC721":
            token_id_t1 = t1.get("tokenId")
            token_id_t3 = t3.get("tokenId")
            passed["cond2"] = (token_id_t1 is not None and token_id_t3 is not None and token_id_t1 == token_id_t3)
            
        elif token_type == "ERC1155":
            is_batch_t1 = t1.get("is_batch", False)
            is_batch_t3 = t3.get("is_batch", False)
            
            if is_batch_t1 and is_batch_t3:
                # Batch transfer: match token_ids arrays
                token_ids_t1 = set(t1.get("token_ids", []))
                token_ids_t3 = set(t3.get("token_ids", []))
                passed["cond2"] = (bool(token_ids_t1) and bool(token_ids_t3) and token_ids_t1 == token_ids_t3)
            elif not is_batch_t1 and not is_batch_t3:
                # Single transfer: match tokenId
                token_id_t1 = t1.get("tokenId")
                token_id_t3 = t3.get("tokenId")
                passed["cond2"] = (token_id_t1 is not None and token_id_t3 is not None and token_id_t1 == token_id_t3)
            else:
                passed["cond2"] = False  # Mixed batch/single transfers

        # Condition 3: Token contract match
        cA1 = t1.get("rawContract", {}).get("address", "").lower()
        cV = t2.get("rawContract", {}).get("address", "").lower()
        cA2 = t3.get("rawContract", {}).get("address", "").lower()
        passed["cond3"] = (cA1 == cV == cA2)

        # Condition 4: Distinct hashes
        hA1 = ea1.get("hash", "")
        hV = ev.get("hash", "")
        hA2 = ea2_candidate.get("hash", "")
        passed["cond4"] = (hA1 != hV and hV != hA2 and hA1 != hA2)

        # Condition 5: Transaction order (flexible version)
        bA1 = hex_to_int(ea1.get("blockNumber", -1))
        bV = hex_to_int(ev.get("blockNumber", -1))
        bA2 = hex_to_int(ea2_candidate.get("blockNumber", -1))

        # Get transaction indices (only relevant for same-block comparisons)
        iA1 = hex_to_int(ea1.get("transactionIndex", -1))
        iV = hex_to_int(ev.get("transactionIndex", -1))
        iA2 = hex_to_int(ea2_candidate.get("transactionIndex", -1))

        # Check overall temporal order: EA1 → EV → EA2
        if bA1 == bV == bA2:
            # All in same block: strict transaction index order required
            passed["cond5"] = (iA1 < iV < iA2)
        elif bA1 == bV and bV < bA2:
            # EA1 and EV in same block, EA2 in later block
            passed["cond5"] = (iA1 < iV)  # EA1 before EV in same block
        elif bA1 < bV and bV == bA2:
            # EA1 in earlier block, EV and EA2 in same block
            passed["cond5"] = (iV < iA2)  # EV before EA2 in same block
        elif bA1 < bV < bA2:
            # All in different blocks: perfect temporal order
            passed["cond5"] = True
        else:
            # Any other ordering is invalid for insertion attack
            passed["cond5"] = False

        # Condition 6: Gas price order
        gA1 = hex_to_int(ea1.get("gasPrice", 0))
        gV = hex_to_int(ev.get("gasPrice", 0))
        gA2 = hex_to_int(ea2_candidate.get("gasPrice", 0))
        passed["cond6"] = (gA1 > gV and gV >= gA2)

        # Update stats
        passed_count = sum(passed.values())
        for i in range(1, 7):
            condition_stats[f"cond{i}"]["pass" if passed[f"cond{i}"] else "fail"] += 1

        # Store the results in the entry for analysis
        entry["flow_consistency"] = passed["cond1"]
        entry["token_identifier_match"] = passed["cond2"]
        entry["token_contract_match"] = passed["cond3"]
        entry["distinct_hashes"] = passed["cond4"]
        entry["tx_order_correct"] = passed["cond5"]
        entry["gas_price_order_correct"] = passed["cond6"]
        entry["matched_token_type"] = token_type

        return int(all(passed.values())), passed_count

    except Exception as e:
        failure_reasons[f"exception_{type(e).__name__}"] += 1
        return 0, 0

def is_suppression(entry):
    """Determine if an entry is a suppression attack using heuristics and add fields"""
    cluster_size = 0
    min_gas_used = 0
    min_gas_util_ratio = 0.0
    victim_before = 0
    victim_after = 0
    heuristic_1 = 0  # Cluster size > 1
    heuristic_2 = 0  # All gas > 21k
    heuristic_3 = 0  # All gas util > 99%
    
    try:
        transactions = entry.get("attacker_transactions", [])
        cluster_size = len(transactions)
        
        # Heuristic 1: Cluster must have more than one transaction
        heuristic_1 = 1 if cluster_size > 1 else 0
        
        # Only check other heuristics if cluster is valid
        if heuristic_1:
            gas_utils = []
            gas_used_list = []
            heuristic_2 = 1  # Start assuming all meet criteria
            heuristic_3 = 1  # Start assuming all meet criteria
            
            for tx in transactions:
                gas_used = tx.get("gasUsed", 0)
                gas_limit = tx.get("gas", 1)  # prevent division by zero
                ratio = gas_used / gas_limit
                
                gas_used_list.append(gas_used)
                gas_utils.append(ratio)
                
                # Heuristic 2: Gas used > 21,000
                if gas_used <= 21000:
                    heuristic_2 = 0
                
                # Heuristic 3: Gas used/limit ratio > 99%
                if ratio <= 0.99:
                    heuristic_3 = 0
            
            min_gas_used = min(gas_used_list) if gas_used_list else 0
            min_gas_util_ratio = min(gas_utils) if gas_utils else 0.0
            
            # Get victim transaction index using robust method
            victim_idx = get_victim_transaction_index(entry.get("victim_details", {}))
            
            # Get all valid transaction indices from attacker transactions
            tx_idxs = []
            for tx in transactions:
                idx = tx.get("transactionIndex", -1)
                if isinstance(idx, int) and idx >= 0:
                    tx_idxs.append(idx)
            
            victim_before = sum(1 for idx in tx_idxs if idx < victim_idx)
            victim_after = sum(1 for idx in tx_idxs if idx > victim_idx)
        
        # Suppression label requires all 3 heuristics to pass
        suppression_label = 1 if (heuristic_1 and heuristic_2 and heuristic_3) else 0
        
        return suppression_label, cluster_size, min_gas_used, min_gas_util_ratio, victim_before, victim_after, heuristic_1, heuristic_2, heuristic_3
        
    except (KeyError, TypeError) as e:
        print(f"Error processing entry: {e}")
        return 0, cluster_size, min_gas_used, min_gas_util_ratio, victim_before, victim_after, 0, 0, 0

def label_displacement_dataset(input_path, output_path):
    """Process all entries and add displacement labels + heuristic fields"""
    print(f"Loading dataset from {input_path}")
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    total_entries = len(dataset)
    print(f"Processing {total_entries} entries...")
    displacement_count = 0

    for i, entry in enumerate(dataset):
        # Add displacement label and heuristic fields
        displacement_label, cond1, cond2, similarity_ratio = is_displacement(entry)
        entry["displacement_label"] = displacement_label
        entry["addresses_distinct"] = cond1
        entry["attacker_gas_higher"] = cond2
        entry["bytecode_similarity"] = similarity_ratio

        displacement_count += displacement_label

        if (i + 1) % 1000 == 0 or (i + 1) == total_entries:
            print(f"  Processed {i+1}/{total_entries} entries")

    print(f"Saving labeled dataset to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Done! Displacement attacks identified: {displacement_count}/{total_entries}")
    print(f"Displacement rate: {displacement_count/total_entries*100:.2f}%")
    
    return dataset, displacement_count

def label_insertion_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        dataset = json.load(f)

    stats = {
        'total': len(dataset),
        'passed': 0,
        'failed': 0,
        'failure_reasons': defaultdict(int),
        'condition_stats': {f"cond{i}": {'pass': 0, 'fail': 0} for i in range(1, 7)}
    }

    for entry in dataset:
        label, passed_conditions = is_insertion(entry, stats['condition_stats'], stats['failure_reasons'])
        entry["insertion_label"] = label
        
        if label:
            stats['passed'] += 1
        else:
            stats['failed'] += 1
            if passed_conditions > 0:
                stats['failure_reasons'][f'failed_{passed_conditions}_conditions'] += 1

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print("\n=== Labeling Results ===")
    print(f"Total entries: {stats['total']}")
    print(f"Passed (insertion attacks): {stats['passed']} ({stats['passed']/stats['total']*100:.2f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.2f}%)")

    print("\n=== Condition Pass/Fail Rates ===")
    for cond in sorted(stats['condition_stats'].keys()):
        pass_count = stats['condition_stats'][cond]['pass']
        total = stats['total']
        print(f"{cond}: Pass={pass_count} ({pass_count/total*100:.2f}%), Fail={total-pass_count} ({(total-pass_count)/total*100:.2f}%)")
    
    return dataset, stats

def label_suppression_dataset(input_path, output_path):
    """Process all entries and add suppression labels + heuristic fields"""
    print(f"Loading dataset from {input_path}")
    with open(input_path, 'r') as f:
        dataset = json.load(f)
    
    suppression_count = 0
    total_entries = len(dataset)
    print(f"Processing {total_entries} entries...")

    for i, entry in enumerate(dataset):
        # Get suppression label and heuristic values
        result = is_suppression(entry)
        suppression_label = result[0]
        cluster_size = result[1]
        min_gas_used = result[2]
        min_gas_util_ratio = result[3]
        victim_before = result[4]
        victim_after = result[5]
        heuristic_1 = result[6]
        heuristic_2 = result[7]
        heuristic_3 = result[8]
        
        # Add all fields to the entry
        entry.update({
            "suppression_label": suppression_label,
            "cluster_size": cluster_size,
            "min_gas_used": min_gas_used,
            "min_gas_util_ratio": min_gas_util_ratio,
            "ClusterSize>1": heuristic_1,
            "AllGas>21k": heuristic_2,
            "GasUtil>99%": heuristic_3
        })

        suppression_count += suppression_label
        
        if (i + 1) % 1000 == 0 or (i + 1) == total_entries:
            print(f"  Processed {i+1}/{total_entries} entries")
    
    print(f"Saving labeled dataset to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Done! Suppression attacks identified: {suppression_count}/{total_entries}")
    print(f"Suppression rate: {suppression_count/total_entries:.2%}")
    
    return dataset, suppression_count

def run_complete_pipeline():
    """Run the complete attack detection pipeline"""
    print("=" * 60)
    print("STARTING COMPLETE ATTACK DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Displacement Attack Detection
    print("\n" + "=" * 40)
    print("STEP 1: DETECTING DISPLACEMENT ATTACKS")
    print("=" * 40)
    dataset, displacement_count = label_displacement_dataset(INPUT_FILE, DIS_OUTPUT_FILE)
    
    # Step 2: Insertion Attack Detection
    print("\n" + "=" * 40)
    print("STEP 2: DETECTING INSERTION ATTACKS")
    print("=" * 40)
    dataset, insertion_stats = label_insertion_dataset(DIS_OUTPUT_FILE, INS_OUTPUT_FILE)
    
    # Step 3: Suppression Attack Detection
    print("\n" + "=" * 40)
    print("STEP 3: DETECTING SUPPRESSION ATTACKS")
    print("=" * 40)
    dataset, suppression_count = label_suppression_dataset(INS_OUTPUT_FILE, SUP_OUTPUT_FILE)
    
    # Save final combined dataset
    print(f"\nSaving final combined dataset to {FINAL_OUTPUT_FILE}")
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Summary Report
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - SUMMARY REPORT")
    print("=" * 60)
    total_entries = len(dataset)
    print(f"Total entries processed: {total_entries}")
    print(f"Displacement attacks: {displacement_count} ({displacement_count/total_entries*100:.2f}%)")
    print(f"Insertion attacks: {insertion_stats['passed']} ({insertion_stats['passed']/total_entries*100:.2f}%)")
    print(f"Suppression attacks: {suppression_count} ({suppression_count/total_entries*100:.2f}%)")
    
    # Check for multi-label entries
    multi_label_count = 0
    for entry in dataset:
        labels = [
            entry.get("displacement_label", 0),
            entry.get("insertion_label", 0),
            entry.get("suppression_label", 0)
        ]
        if sum(labels) > 1:
            multi_label_count += 1
    
    print(f"Entries with multiple attack labels: {multi_label_count}")
    print(f"Final dataset saved to: {FINAL_OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    run_complete_pipeline()