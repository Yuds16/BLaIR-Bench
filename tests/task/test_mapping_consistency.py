#!/usr/bin/env python3

"""
Test script to verify that seq_rec and cf generate identical item2id mappings
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from blair.dataset.amazon_utils import remap_id
# from blair.cf.dataset.process_amazon_2023 import remap_id_cf

def create_mock_datasets():
    """Create mock datasets that simulate seq_rec and cf data structures"""
    
    # Mock seq_rec datasets structure
    seq_rec_datasets = {
        'train': {
            'user_id': ['u1', 'u2', 'u3'],
            'parent_asin': ['B123', 'B456', 'B789'], 
            'history': ['B111 B222', 'B333', 'B456 B123']
        },
        'valid': {
            'user_id': ['u4', 'u5'],
            'parent_asin': ['B333', 'B111'],
            'history': ['B789', 'B222 B456']
        },
        'test': {
            'user_id': ['u6'],
            'parent_asin': ['B222'],
            'history': ['B111 B333 B789']
        }
    }
    
    # Mock cf dataframes (shuffled order to simulate real difference)
    # Using simple dict structure instead of pandas DataFrame
    cf_dfs = [
        {'user_id': ['u2', 'u1', 'u3'], 'parent_asin': ['B456', 'B123', 'B789']},
        {'user_id': ['u5', 'u4'], 'parent_asin': ['B111', 'B333']},
        {'user_id': ['u6'], 'parent_asin': ['B222']}
    ]
    
    return seq_rec_datasets, cf_dfs

# def test_mapping_consistency():
#     """Test that both mapping functions produce identical item2id mappings"""
#     print("Testing mapping consistency between seq_rec and cf...")
    
#     # Create mock data
#     seq_rec_datasets, cf_dfs = create_mock_datasets()
    
#     # Generate mappings
#     seq_rec_maps = remap_id(seq_rec_datasets)
#     cf_maps = remap_id_cf(cf_dfs)
    
#     # Compare item2id mappings
#     print("\nseq_rec item2id mapping:")
#     for item, id_ in sorted(seq_rec_maps['item2id'].items()):
#         print(f"  {item}: {id_}")
    
#     print("\ncf item2id mapping:")
#     for item, id_ in sorted(cf_maps['item2id'].items()):
#         print(f"  {item}: {id_}")
    
#     # Check if identical
#     items_match = seq_rec_maps['item2id'] == cf_maps['item2id']
#     print(f"\nItem mappings identical: {items_match}")
    
#     if not items_match:
#         print("Differences found:")
#         all_items = set(seq_rec_maps['item2id'].keys()) | set(cf_maps['item2id'].keys())
#         for item in sorted(all_items):
#             seq_id = seq_rec_maps['item2id'].get(item, 'MISSING')
#             cf_id = cf_maps['item2id'].get(item, 'MISSING')
#             if seq_id != cf_id:
#                 print(f"  {item}: seq_rec={seq_id}, cf={cf_id}")
    
#     return items_match

if __name__ == "__main__":
    # success = test_mapping_consistency()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")