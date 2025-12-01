#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script validate toàn bộ hierarchy"""

import json
from collections import defaultdict

# Đọc chunks
with open('output_admission/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Tạo map ID -> chunk
chunk_map = {c['chunk_id']: c for c in chunks}

print("\n" + "="*80)
print("VALIDATION REPORT - HIERARCHY STRUCTURE")
print("="*80 + "\n")

# 1. Kiểm tra parent-child consistency
print("1. KIỂM TRA PARENT-CHILD CONSISTENCY")
print("-" * 80)

errors = []
warnings = []

for chunk in chunks:
    chunk_id = chunk['chunk_id']
    parent_id = chunk['metadata']['parent_id']
    children_ids = chunk['metadata']['children_ids']
    section_type = chunk['metadata']['section_type']
    section_code = chunk['metadata']['section_code']

    # Kiểm tra parent tồn tại
    if parent_id and parent_id not in chunk_map:
        errors.append(f"✗ {section_code}: Parent ID không tồn tại: {parent_id}")

    # Kiểm tra children tồn tại
    for child_id in children_ids:
        if child_id not in chunk_map:
            errors.append(f"✗ {section_code}: Child ID không tồn tại: {child_id}")
        else:
            # Kiểm tra child có trỏ lại parent không
            child = chunk_map[child_id]
            if child['metadata']['parent_id'] != chunk_id:
                errors.append(f"✗ {section_code}: Child {child['metadata']['section_code']} không trỏ lại parent")

if errors:
    for err in errors[:10]:
        print(err)
    if len(errors) > 10:
        print(f"... và {len(errors) - 10} lỗi khác")
else:
    print("✓ Tất cả parent-child relationships đều hợp lệ")

print()

# 2. Kiểm tra level consistency
print("2. KIỂM TRA LEVEL CONSISTENCY")
print("-" * 80)

level_errors = []

hierarchy_levels = {
    'root': 1,
    'chuong': 2,
    'muc': 3,
    'dieu': 3,  # Có thể = 3 (trực tiếp dưới chương) hoặc 4 (dưới mục)
    'khoan': 4,  # Có thể = 4 hoặc 5
    'item_abc': 5,  # Có thể = 5 hoặc 6
    'item_dash': 6,  # Có thể = 6 hoặc 7
    'item_plus': 7,  # Có thể = 7 hoặc 8
}

for chunk in chunks:
    section_type = chunk['metadata']['section_type']
    level = chunk['metadata']['level']
    section_code = chunk['metadata']['section_code']
    parent_id = chunk['metadata']['parent_id']

    # Kiểm tra level với parent
    if parent_id:
        parent = chunk_map[parent_id]
        parent_level = parent['metadata']['level']

        if level != parent_level + 1:
            level_errors.append(
                f"✗ {section_code} (level {level}): "
                f"Parent {parent['metadata']['section_code']} có level {parent_level}, "
                f"expected level {parent_level + 1}"
            )

if level_errors:
    for err in level_errors[:10]:
        print(err)
    if len(level_errors) > 10:
        print(f"... và {len(level_errors) - 10} lỗi khác")
else:
    print("✓ Tất cả levels đều hợp lệ")

print()

# 3. Thống kê cấu trúc
print("3. THỐNG KÊ CẤU TRÚC")
print("-" * 80)

# Đếm số lượng theo type
type_counts = defaultdict(int)
for chunk in chunks:
    type_counts[chunk['metadata']['section_type']] += 1

print(f"Tổng số chunks: {len(chunks)}")
for stype in sorted(type_counts.keys()):
    print(f"  - {stype:12s}: {type_counts[stype]:4d} chunks")

print()

# Đếm số lượng theo Chương
print("Số lượng chunks theo Chương:")
chuong_chunks = defaultdict(int)
for chunk in chunks:
    module = chunk['metadata']['module']
    chuong_chunks[module] += 1

for module in sorted(chuong_chunks.keys()):
    print(f"  - {module:15s}: {chuong_chunks[module]:4d} chunks")

print()

# 4. Kiểm tra section_code format
print("4. KIỂM TRA SECTION CODE FORMAT")
print("-" * 80)

code_errors = []

for chunk in chunks:
    section_code = chunk['metadata']['section_code']
    section_type = chunk['metadata']['section_type']
    parent_id = chunk['metadata']['parent_id']

    # Root không cần kiểm tra
    if section_type == 'root':
        continue

    # Section code phải bắt đầu bằng parent code
    if parent_id:
        parent = chunk_map[parent_id]
        parent_code = parent['metadata']['section_code']

        # Nếu parent là root, bỏ qua
        if parent['metadata']['section_type'] == 'root':
            continue

        # Section code phải bắt đầu bằng parent_code
        if not section_code.startswith(parent_code + '.'):
            code_errors.append(
                f"✗ {section_code}: Không match với parent code {parent_code}"
            )

if code_errors:
    for err in code_errors[:10]:
        print(err)
    if len(code_errors) > 10:
        print(f"... và {len(code_errors) - 10} lỗi khác")
else:
    print("✓ Tất cả section codes đều hợp lệ")

print()

# 5. Kiểm tra siblings
print("5. KIỂM TRA SIBLING RELATIONSHIPS")
print("-" * 80)

sibling_errors = []

for chunk in chunks:
    chunk_id = chunk['chunk_id']
    sibling_ids = chunk['metadata']['sibling_ids']
    parent_id = chunk['metadata']['parent_id']
    section_code = chunk['metadata']['section_code']

    # Kiểm tra tất cả siblings có cùng parent không
    for sibling_id in sibling_ids:
        if sibling_id not in chunk_map:
            sibling_errors.append(f"✗ {section_code}: Sibling ID không tồn tại: {sibling_id}")
        else:
            sibling = chunk_map[sibling_id]
            if sibling['metadata']['parent_id'] != parent_id:
                sibling_errors.append(
                    f"✗ {section_code}: Sibling {sibling['metadata']['section_code']} "
                    f"không cùng parent"
                )

if sibling_errors:
    for err in sibling_errors[:10]:
        print(err)
    if len(sibling_errors) > 10:
        print(f"... và {len(sibling_errors) - 10} lỗi khác")
else:
    print("✓ Tất cả sibling relationships đều hợp lệ")

print()

# Tổng kết
print("="*80)
print("TỔNG KẾT VALIDATION")
print("="*80)

total_errors = len(errors) + len(level_errors) + len(code_errors) + len(sibling_errors)

if total_errors == 0:
    print("✅ HOÀN HẢO! Không có lỗi nào được phát hiện.")
    print(f"   Tổng số chunks: {len(chunks)}")
    print(f"   Hierarchy graph: hợp lệ 100%")
else:
    print(f"❌ Phát hiện {total_errors} lỗi:")
    print(f"   - Parent-child errors: {len(errors)}")
    print(f"   - Level errors: {len(level_errors)}")
    print(f"   - Section code errors: {len(code_errors)}")
    print(f"   - Sibling errors: {len(sibling_errors)}")

print("="*80 + "\n")
