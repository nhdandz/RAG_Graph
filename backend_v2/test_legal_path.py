"""
Test script to verify legal path formatting
"""
import json

def build_legal_hierarchy_path(chunk, chunk_map):
    """Test version of build_legal_hierarchy_path
    Thứ tự: Chương > Mục > Điều > Khoản > Điểm
    """
    path_parts = []
    current_chunk = chunk
    visited = set()

    while current_chunk:
        chunk_id = current_chunk.get('chunk_id')
        if chunk_id in visited:
            break
        visited.add(chunk_id)

        metadata = current_chunk.get('metadata', {})
        section_type = metadata.get('section_type', '')
        section_code = metadata.get('section_code', '')
        section_title = metadata.get('section_title', '')

        type_labels = {
            'chuong': 'Chương',
            'muc': 'Mục',
            'dieu': 'Điều',
            'khoan': 'Khoản',
            'diem': 'Điểm',
            'item_abc': 'Điểm'
        }

        label = type_labels.get(section_type, '')

        if label:
            code_parts = section_code.split('.')

            # Parse theo cấu trúc: Chương.Mục.Điều.Khoản.Điểm
            # Ví dụ: VI.2.48.4 = Chương VI, Mục 2, Điều 48, Khoản 4

            if section_type == 'chuong' and len(code_parts) >= 1:
                path_parts.insert(0, f'Chương {code_parts[0]}')
            elif section_type == 'muc' and len(code_parts) >= 2:
                path_parts.insert(0, f'Mục {code_parts[1]}')
            elif section_type == 'dieu' and len(code_parts) >= 3:
                path_parts.insert(0, f'Điều {code_parts[2]}')
            elif section_type == 'khoan' and len(code_parts) >= 4:
                path_parts.insert(0, f'Khoản {code_parts[3]}')
            elif section_type in ['diem', 'item_abc'] and len(code_parts) >= 5:
                path_parts.insert(0, f'Điểm {code_parts[4]}')

        parent_id = metadata.get('parent_id')
        if parent_id and parent_id in chunk_map:
            current_chunk = chunk_map[parent_id]
        else:
            break

    return ' > '.join(path_parts) if path_parts else section_title


if __name__ == "__main__":
    # Load chunks
    print("Loading chunks...")
    with open('output_admission/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Create chunk_map
    chunk_map = {chunk['chunk_id']: chunk for chunk in chunks}

    print(f"✓ Loaded {len(chunks)} chunks\n")

    # Test different types
    print("=" * 80)
    print("TESTING LEGAL PATH FORMATTING")
    print("=" * 80)

    test_cases = [
        {"name": "Root", "index": 0},
        {"name": "Chương", "index": 1},
        {"name": "Điều", "index": 2},
        {"name": "Khoản", "index": 10},
        {"name": "Khoản khác", "index": 20},
    ]

    for test in test_cases:
        i = test["index"]
        if i < len(chunks):
            chunk = chunks[i]
            meta = chunk['metadata']
            legal_path = build_legal_hierarchy_path(chunk, chunk_map)
            title_path = ' > '.join(meta.get('title_path', []))

            print(f"\n{'─' * 80}")
            print(f"Test: {test['name']}")
            print(f"{'─' * 80}")
            print(f"Section Type : {meta.get('section_type')}")
            print(f"Section Code : {meta.get('section_code')}")
            print(f"Section Title: {meta.get('section_title')[:60]}...")
            print(f"\n✓ Legal Path : {legal_path}")
            print(f"  Title Path : {title_path[:70]}...")

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)
