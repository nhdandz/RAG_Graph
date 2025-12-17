# -*- coding: utf-8 -*-
"""
Script test chunking cho Tendoo documentation
"""

import os
import json
from tendoo_rag_chunking import TendooDocumentChunker


def test_chunking():
    """Test chunking với file mẫu"""

    print("="*80)
    print("TEST TENDOO CHUNKING")
    print("="*80)

    # Đường dẫn file
    DOCX_PATH = "tendoo_guide.docx"
    OUTPUT_DIR = "output_tendoo"

    # Kiểm tra file tồn tại
    if not os.path.exists(DOCX_PATH):
        print(f"\n❌ Không tìm thấy file {DOCX_PATH}")
        print("Đang tạo file mẫu...")
        os.system("python create_tendoo_sample.py")
        print()

    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Khởi tạo chunker
    chunker = TendooDocumentChunker()

    # Xử lý tài liệu
    print("\n" + "="*80)
    print("BƯỚC 1: PARSE VÀ CHUNKING")
    print("="*80)
    chunks = chunker.process_document(DOCX_PATH)

    # Lưu kết quả
    print("\n" + "="*80)
    print("BƯỚC 2: LƯU KẾT QUẢ")
    print("="*80)
    chunker.save_chunks(os.path.join(OUTPUT_DIR, 'chunks.json'))
    chunker.save_graph(OUTPUT_DIR)

    # In tóm tắt
    print("\n" + "="*80)
    print("BƯỚC 3: THỐNG KÊ")
    print("="*80)
    chunker.print_summary()

    # In mẫu chunks
    print("\n" + "="*80)
    print("BƯỚC 4: XEM MẪU CHUNKS")
    print("="*80)
    chunker.print_sample_chunks(5)

    # Phân tích chi tiết
    analyze_chunks(chunks)

    # Test tìm kiếm theo hierarchy
    test_hierarchy_search(chunks)

    print("\n" + "="*80)
    print("✅ TEST HOÀN THÀNH!")
    print("="*80)
    print(f"\nKết quả đã được lưu tại: {OUTPUT_DIR}/")
    print(f"  - chunks.json: {len(chunks)} chunks")
    print(f"  - hierarchy_graph.gexf: đồ thị phân cấp")


def analyze_chunks(chunks):
    """Phân tích chi tiết chunks"""
    print("\n" + "="*80)
    print("PHÂN TÍCH CHI TIẾT")
    print("="*80)

    # Phân tích theo level
    print("\n📊 Phân bố theo level:")
    from collections import defaultdict
    level_counts = defaultdict(int)
    for chunk in chunks:
        level_counts[chunk.metadata.level] += 1

    for level in sorted(level_counts.keys()):
        print(f"  Level {level}: {level_counts[level]} chunks")

    # Tìm chunk sâu nhất
    max_level = max([c.metadata.level for c in chunks])
    deepest = [c for c in chunks if c.metadata.level == max_level]
    print(f"\n🔍 Chunk sâu nhất (level {max_level}):")
    for c in deepest[:3]:
        print(f"  - {c.metadata.section_code}: {c.metadata.section_title[:50]}...")

    # Phân tích title path
    print("\n📂 Một số title paths mẫu:")
    for chunk in chunks[1:6]:  # Bỏ qua root
        path = " > ".join(chunk.metadata.title_path)
        print(f"  {chunk.metadata.section_code}: {path[:100]}...")

    # Phân tích tags
    print("\n🏷️ Top tags:")
    tag_counts = defaultdict(int)
    for chunk in chunks:
        for tag in chunk.metadata.tags:
            tag_counts[tag] += 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in sorted_tags[:10]:
        print(f"  - {tag}: {count} chunks")


def test_hierarchy_search(chunks):
    """Test tìm kiếm theo hierarchy"""
    print("\n" + "="*80)
    print("TEST TÌM KIẾM THEO HIERARCHY")
    print("="*80)

    # Test 1: Tìm tất cả sections level 1
    print("\n🔍 Test 1: Tìm sections level 1")
    level1_sections = [c for c in chunks if c.metadata.section_type == 'section_1']
    print(f"Tìm thấy {len(level1_sections)} sections level 1:")
    for section in level1_sections:
        print(f"  - {section.metadata.section_code}: {section.metadata.section_title}")

    # Test 2: Tìm children của section 1
    print("\n🔍 Test 2: Tìm children của section '1'")
    section_1 = [c for c in chunks if c.metadata.section_code == '1']
    if section_1:
        chunk_map = {c.chunk_id: c for c in chunks}
        section = section_1[0]
        print(f"Section: {section.metadata.section_title}")
        print(f"Có {len(section.metadata.children_ids)} children:")
        for child_id in section.metadata.children_ids[:5]:  # Chỉ hiển thị 5 đầu
            child = chunk_map.get(child_id)
            if child:
                print(f"  - {child.metadata.section_code}: {child.metadata.section_title[:50]}...")

    # Test 3: Tìm siblings của section 1.1
    print("\n🔍 Test 3: Tìm siblings của section '1.1'")
    section_11 = [c for c in chunks if c.metadata.section_code == '1.1']
    if section_11:
        chunk_map = {c.chunk_id: c for c in chunks}
        section = section_11[0]
        print(f"Section: {section.metadata.section_title}")
        print(f"Có {len(section.metadata.sibling_ids)} siblings:")
        for sibling_id in section.metadata.sibling_ids:
            sibling = chunk_map.get(sibling_id)
            if sibling:
                print(f"  - {sibling.metadata.section_code}: {sibling.metadata.section_title[:50]}...")

    # Test 4: Tìm tất cả chunks về "hóa đơn"
    print("\n🔍 Test 4: Tìm chunks có tag 'hóa đơn'")
    invoice_chunks = [c for c in chunks if 'hóa đơn' in c.metadata.tags]
    print(f"Tìm thấy {len(invoice_chunks)} chunks:")
    for chunk in invoice_chunks[:5]:
        print(f"  - {chunk.metadata.section_code}: {chunk.metadata.section_title[:60]}...")

    # Test 5: Tìm chunks về "thanh toán"
    print("\n🔍 Test 5: Tìm chunks có tag 'thanh toán'")
    payment_chunks = [c for c in chunks if 'thanh toán' in c.metadata.tags]
    print(f"Tìm thấy {len(payment_chunks)} chunks:")
    for chunk in payment_chunks[:5]:
        print(f"  - {chunk.metadata.section_code}: {chunk.metadata.section_title[:60]}...")


def print_chunk_tree(chunks, max_depth=3):
    """In cây phân cấp của chunks"""
    print("\n" + "="*80)
    print("CÂY PHÂN CẤP CHUNKS")
    print("="*80)

    # Tạo map
    chunk_map = {c.chunk_id: c for c in chunks}

    # Tìm root
    root_chunks = [c for c in chunks if c.metadata.parent_id is None]

    def print_node(chunk, depth=0):
        if depth > max_depth:
            return

        indent = "  " * depth
        symbol = "├─" if depth > 0 else ""
        title = chunk.metadata.section_title[:60]
        print(f"{indent}{symbol} [{chunk.metadata.section_code}] {title}")

        # In children
        for child_id in chunk.metadata.children_ids[:5]:  # Giới hạn 5 children
            child = chunk_map.get(child_id)
            if child:
                print_node(child, depth + 1)

        if len(chunk.metadata.children_ids) > 5:
            print(f"{'  ' * (depth + 1)}... và {len(chunk.metadata.children_ids) - 5} mục khác")

    for root in root_chunks:
        print_node(root)


if __name__ == "__main__":
    test_chunking()
