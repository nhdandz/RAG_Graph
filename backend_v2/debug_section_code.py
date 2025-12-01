#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug section code issues"""

import json

# Đọc chunks
with open('output_admission/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Tạo map ID -> chunk
chunk_map = {c['chunk_id']: c for c in chunks}

# Tìm chunk có section_code "III.1.8"
print("\n" + "="*80)
print("DEBUG: III.1.8")
print("="*80 + "\n")

target = None
for chunk in chunks:
    if chunk['metadata']['section_code'] == 'III.1.8':
        target = chunk
        break

if target:
    print(f"Chunk: {target['metadata']['section_code']}")
    print(f"Type: {target['metadata']['section_type']}")
    print(f"Number: {target['metadata']['section_number']}")
    print(f"Title: {target['metadata']['section_title']}")
    print(f"Level: {target['metadata']['level']}")
    print(f"Content: {target['content'][:100]}")

    # Trace parent chain
    print(f"\nParent chain:")
    current = target
    depth = 0
    while current and depth < 10:
        parent_id = current['metadata']['parent_id']
        if not parent_id:
            break

        parent = chunk_map.get(parent_id)
        if not parent:
            print(f"  {depth+1}. PARENT NOT FOUND: {parent_id}")
            break

        print(f"  {depth+1}. {parent['metadata']['section_code']:20s} ({parent['metadata']['section_type']})")
        current = parent
        depth += 1

    # Trace children
    print(f"\nChildren IDs:")
    if target['metadata']['children_ids']:
        for child_id in target['metadata']['children_ids'][:5]:
            child = chunk_map.get(child_id)
            if child:
                print(f"  - {child['metadata']['section_code']:20s} ({child['metadata']['section_type']})")

print("\n" + "="*80)
print("DEBUG: VIII.1")
print("="*80 + "\n")

target2 = None
for chunk in chunks:
    if chunk['metadata']['section_code'] == 'VIII.1':
        target2 = chunk
        break

if target2:
    print(f"Chunk: {target2['metadata']['section_code']}")
    print(f"Type: {target2['metadata']['section_type']}")
    print(f"Number: {target2['metadata']['section_number']}")
    print(f"Title: {target2['metadata']['section_title']}")
    print(f"Level: {target2['metadata']['level']}")
    print(f"Content: {target2['content'][:100]}")

    # Children
    print(f"\nChildren ({len(target2['metadata']['children_ids'])} total):")
    for child_id in target2['metadata']['children_ids'][:10]:
        child = chunk_map.get(child_id)
        if child:
            print(f"  - {child['metadata']['section_code']:20s} ({child['metadata']['section_type']}) {child['metadata']['section_title'][:30]}")

print("\n" + "="*80)
