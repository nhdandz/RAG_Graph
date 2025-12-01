#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize hierarchy tree"""

import json

# Đọc chunks
with open('output_admission/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Tạo map ID -> chunk
chunk_map = {c['chunk_id']: c for c in chunks}

def print_tree(chunk, depth=0, max_depth=4):
    """In cây phân cấp"""
    if depth > max_depth:
        return

    # Prefix cho mỗi level
    prefix = "  " * depth

    # Icon theo type
    icons = {
        'root': '📁',
        'chuong': '📂',
        'muc': '📋',
        'dieu': '📄',
        'khoan': '▪',
        'item_abc': '•',
        'item_dash': '-',
        'item_plus': '+'
    }

    icon = icons.get(chunk['metadata']['section_type'], '?')
    code = chunk['metadata']['section_code']
    title = chunk['metadata']['section_title'][:60]

    # Chỉ hiện children count nếu có
    children_count = len(chunk['metadata']['children_ids'])
    count_str = f" ({children_count})" if children_count > 0 else ""

    print(f"{prefix}{icon} {code:15s} {title}{count_str}")

    # In children
    if depth < max_depth:
        for child_id in chunk['metadata']['children_ids'][:5]:  # Giới hạn 5 children
            if child_id in chunk_map:
                print_tree(chunk_map[child_id], depth + 1, max_depth)

        if len(chunk['metadata']['children_ids']) > 5:
            print(f"{prefix}  ... và {len(chunk['metadata']['children_ids']) - 5} items khác")

# Tìm ROOT
root = None
for c in chunks:
    if c['metadata']['section_type'] == 'root':
        root = c
        break

print("\n" + "="*80)
print("CẤU TRÚC PHÂN CẤP CỦA TÀI LIỆU TUYỂN SINH")
print("="*80 + "\n")

if root:
    print_tree(root, max_depth=3)

print("\n" + "="*80 + "\n")

# In chi tiết Chương I
print("="*80)
print("CHI TIẾT CHƯƠNG I")
print("="*80 + "\n")

chuong_1 = None
for c in chunks:
    if c['metadata']['section_code'] == 'I':
        chuong_1 = c
        break

if chuong_1:
    print_tree(chuong_1, max_depth=4)

print("\n" + "="*80 + "\n")
