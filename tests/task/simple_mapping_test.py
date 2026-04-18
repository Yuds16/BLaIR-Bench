#!/usr/bin/env python3

# Simple test to verify the sorting logic works
def test_sorting_consistency():
    """Test that sorting produces consistent ordering"""
    
    # Items from seq_rec in some order
    seq_items = {'B789', 'B123', 'B456', 'B111', 'B222', 'B333'}
    
    # Items from cf in different order (simulating shuffled data)
    cf_items = {'B456', 'B111', 'B789', 'B333', 'B222', 'B123'}
    
    # Sort both
    seq_sorted = sorted(seq_items)
    cf_sorted = sorted(cf_items)
    
    print("seq_rec items sorted:", seq_sorted)
    print("cf items sorted:     ", cf_sorted)
    print("Identical ordering:", seq_sorted == cf_sorted)
    
    # Create mappings
    seq_item2id = {'[PAD]': 0}
    cf_item2id = {'[PAD]': 0}
    
    for i, item in enumerate(seq_sorted, 1):
        seq_item2id[item] = i
        
    for i, item in enumerate(cf_sorted, 1):
        cf_item2id[item] = i
    
    print("\nseq_rec mapping:", seq_item2id)
    print("cf mapping:     ", cf_item2id)
    print("Mappings identical:", seq_item2id == cf_item2id)
    
    assert seq_item2id == cf_item2id

if __name__ == "__main__":
    test_sorting_consistency()
