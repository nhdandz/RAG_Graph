#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script kiểm tra hierarchy của chunks"""

import json

# Đọc chunks
with open('output_admission/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Tạo map ID -> chunk
chunk_map = {c['chunk_id']: c for c in chunks}

# Kiểm tra các Điều trong Chương I
print("\n" + "="*80)
print("KIỂM TRA HIERARCHY CỦA CÁC ĐIỀU TRONG CHƯƠNG I")
print("="*80 + "\n")

chuong_1 = None
for chunk in chunks:
    if chunk['metadata']['section_type'] == 'chuong' and chunk['metadata']['section_number'] == 'I':
        chuong_1 = chunk
        break

if chuong_1:
    print(f"✓ Tìm thấy Chương I:")
    print(f"  ID: {chuong_1['chunk_id']}")
    print(f"  Title: {chuong_1['metadata']['section_title']}")
    print(f"  Children: {len(chuong_1['metadata']['children_ids'])} items\n")

    # Lấy các Điều trong Chương I
    dieu_chunks = []
    for chunk in chunks:
        if (chunk['metadata']['section_type'] == 'dieu' and
            chunk['metadata']['section_code'].startswith('I.')):
            dieu_chunks.append(chunk)

    # Sắp xếp theo section_number
    dieu_chunks.sort(key=lambda x: int(x['metadata']['section_number']))

    print(f"Tìm thấy {len(dieu_chunks)} Điều trong Chương I:\n")

    for dieu in dieu_chunks[:10]:  # Chỉ hiện 10 Điều đầu
        parent = chunk_map.get(dieu['metadata']['parent_id'])
        parent_name = parent['metadata']['section_title'] if parent else "None"
        parent_type = parent['metadata']['section_type'] if parent else "None"

        status = "✓" if dieu['metadata']['parent_id'] == chuong_1['chunk_id'] else "✗"

        print(f"{status} Điều {dieu['metadata']['section_number']}: {dieu['metadata']['section_title']}")
        print(f"  Section Code: {dieu['metadata']['section_code']}")
        print(f"  Parent ID: {dieu['metadata']['parent_id']}")
        print(f"  Parent: {parent_type} - {parent_name}")
        print(f"  Level: {dieu['metadata']['level']}")
        print()

# Kiểm tra các Khoản của Điều 3
print("\n" + "="*80)
print("KIỂM TRA HIERARCHY CỦA CÁC KHOẢN TRONG ĐIỀU 3")
print("="*80 + "\n")

dieu_3 = None
for chunk in chunks:
    if (chunk['metadata']['section_type'] == 'dieu' and
        chunk['metadata']['section_number'] == '3' and
        chunk['metadata']['section_code'] == 'I.3'):
        dieu_3 = chunk
        break

if dieu_3:
    print(f"✓ Tìm thấy Điều 3:")
    print(f"  ID: {dieu_3['chunk_id']}")
    print(f"  Title: {dieu_3['metadata']['section_title']}")
    print(f"  Parent: {chunk_map[dieu_3['metadata']['parent_id']]['metadata']['section_title']}")
    print(f"  Children: {len(dieu_3['metadata']['children_ids'])} items\n")

    # Lấy các Khoản trong Điều 3
    khoan_chunks = []
    for chunk in chunks:
        if chunk['metadata']['parent_id'] == dieu_3['chunk_id']:
            khoan_chunks.append(chunk)

    print(f"Các con của Điều 3:\n")

    for khoan in khoan_chunks:
        parent = chunk_map.get(khoan['metadata']['parent_id'])
        parent_name = parent['metadata']['section_title'] if parent else "None"

        status = "✓" if khoan['metadata']['parent_id'] == dieu_3['chunk_id'] else "✗"

        print(f"{status} {khoan['metadata']['section_type']} {khoan['metadata']['section_number']}")
        print(f"  Section Code: {khoan['metadata']['section_code']}")
        print(f"  Content: {khoan['content'][:100]}...")
        print(f"  Parent: {parent_name}")
        print()

print("\n" + "="*80)
