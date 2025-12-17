# -*- coding: utf-8 -*-
"""
Hybrid Hierarchical-Graph Chunking System for Tendoo Documentation
Hệ thống phân tích văn bản hướng dẫn Tendoo theo cấu trúc phân cấp và đồ thị quan hệ
"""

import os
import re
import json
import hashlib
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from docx import Document
import networkx as nx


@dataclass
class SectionMetadata:
    """
    Metadata cho mỗi chunk theo cấu trúc Tendoo Documentation

    Layer 1: Hierarchical structure (section_code)
    Layer 2: Graph relationships (parent/children/siblings/related)
    Layer 3: Metadata (tags, module, titlePath)
    """
    # Layer 1: Hierarchical Structure
    section_code: str  # VD: "1", "1.1", "1.1.1", "1.2.1.1", etc.
    section_type: str  # "root" | "section_1" | "section_2" | "section_3" | "section_4" | "item_number" | "item_dash" | "item_plus"
    section_number: str  # Số thứ tự: "1", "1.1", "1.1.1", etc.
    section_title: str  # Tiêu đề của section

    # Layer 2: Graph Relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)  # Quan hệ ngữ nghĩa

    # Layer 3: Rich Metadata
    title_path: List[str] = field(default_factory=list)  # Đường dẫn đầy đủ từ root
    module: str = ""  # Module/Chương lớn mà section thuộc về
    tags: List[str] = field(default_factory=list)  # Các tags tự động gán

    # Additional Info
    level: int = 0  # Độ sâu trong cây phân cấp
    position: int = 0  # Vị trí trong văn bản
    word_count: int = 0
    is_global_context: bool = False  # Phần đầu là global context


@dataclass
class Chunk:
    """
    Chunk = {
        content: nội dung text
        metadata: SectionMetadata với 3 layers
    }
    """
    chunk_id: str
    content: str
    metadata: SectionMetadata

    def to_dict(self) -> Dict:
        """Chuyển sang dictionary để lưu JSON"""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'metadata': asdict(self.metadata)
        }


class TendooDocumentChunker:
    """
    Parser chuyên biệt cho văn bản hướng dẫn Tendoo

    Cấu trúc phân cấp:
    - Root (Phần đầu)
    - Section level 1: 1., 2., 3., ...
    - Section level 2: 1.1., 1.2., 1.3., ...
    - Section level 3: 1.1.1., 1.1.2., 1.1.3., ...
    - Section level 4+: 1.2.1.1., 1.2.1.2., ...
    - Items trong mục: số (1, 2, 3), dấu -, dấu +
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.hierarchy_graph = nx.DiGraph()  # Đồ thị phân cấp
        self.semantic_graph = nx.Graph()  # Đồ thị ngữ nghĩa

        # Stack để track hierarchy hiện tại
        self.hierarchy_stack: List[Tuple[str, str, int, str]] = []  # (type, number, level, chunk_id)

        # Global context tracking
        self.global_context_content: List[str] = []
        self.found_first_section = False

    def _generate_chunk_id(self, section_code: str, text: str, position: int = 0) -> str:
        """Tạo ID duy nhất cho chunk"""
        hash_input = f"{section_code}_{text[:50]}_{position}".encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()[:16]

    def _detect_section_type(self, text: str) -> Optional[Dict]:
        """
        Phát hiện loại section và trích xuất thông tin

        Returns:
            {
                'type': str,
                'number': str,
                'title': str,
                'full_text': str,
                'level': int  # Độ sâu của section (số dấu chấm)
            }
        """
        text = text.strip()

        # Loại bỏ số trang ở cuối (nếu có) - thường có tab trước số trang
        # VD: "1.1. Cửa hàng	8" -> "1.1. Cửa hàng"
        text = re.sub(r'\t+\d+$', '', text).strip()

        # Pattern cho section với số phân cấp (1., 1.1., 1.1.1., 1.2.1.1., etc.)
        # Phải là đầu dòng và theo sau bởi khoảng trắng hoặc tab
        section_pattern = r'^((?:\d+\.)+)\s+(.+)$'
        match = re.match(section_pattern, text)
        if match:
            section_number = match.group(1).rstrip('.')  # "1.1.1" (bỏ dấu chấm cuối)
            section_title = match.group(2).strip()

            # Đếm độ sâu (số dấu chấm)
            level = section_number.count('.') + 1

            # Xác định type dựa vào level
            section_type = f"section_{level}"

            return {
                'type': section_type,
                'number': section_number,
                'title': section_title,
                'full_text': text,
                'level': level
            }

        # Pattern cho các item đánh số trong mục (không có dấu chấm sau số)
        # VD: "1 Mô tả", "2 Hướng dẫn", etc.
        # Chỉ chấp nhận số 1-2 chữ số để tránh nhầm với năm tháng
        item_number_pattern = r'^(\d{1,2})\s+(.+)$'
        match = re.match(item_number_pattern, text)
        if match and not text[len(match.group(1))].isdigit():  # Đảm bảo không phải số dài
            return {
                'type': 'item_number',
                'number': match.group(1),
                'title': match.group(2).strip(),  # Lấy phần text sau số
                'full_text': text,
                'level': 99  # Level cao để item nằm dưới section
            }

        # Pattern cho các ý gạch ngang -
        dash_pattern = r'^[-–—]\s+(.+)$'
        match = re.match(dash_pattern, text)
        if match:
            return {
                'type': 'item_dash',
                'number': '-',
                'title': match.group(1).strip(),  # Lấy phần text sau dấu -
                'full_text': text,
                'level': 100  # Level cao hơn item_number
            }

        # Pattern cho các ý dấu cộng +
        plus_pattern = r'^\+\s+(.+)$'
        match = re.match(plus_pattern, text)
        if match:
            return {
                'type': 'item_plus',
                'number': '+',
                'title': match.group(1).strip(),  # Lấy phần text sau dấu +
                'full_text': text,
                'level': 101  # Level cao nhất
            }

        return None

    def _build_section_code(self, current_section_number: str, current_section_type: str) -> str:
        """
        Xây dựng section_code

        Logic:
        - Nếu là section (có dạng "1.1.1"): dùng trực tiếp section_number
        - Nếu là item (-, +, số đơn): lấy section_code của parent gần nhất + current

        VD: "1.1.3" hoặc "1.1.3.1" hoặc "1.1.3.1.-" hoặc "1.1.3.1.-.+"
        """
        # Nếu là section với số phân cấp, dùng trực tiếp
        if current_section_type.startswith('section_'):
            return current_section_number

        # Nếu là item, tìm parent section gần nhất
        parent_section_code = None
        for stack_type, stack_number, stack_level, stack_chunk_id in reversed(self.hierarchy_stack[:-1]):
            if stack_type.startswith('section_'):
                # Tìm section parent gần nhất
                parent_section_code = stack_number
                break
            elif stack_type != 'root':
                # Nếu là item, cần lấy code của nó
                # Tìm trong chunks đã tạo
                parent_chunks = [c for c in self.chunks if c.chunk_id == stack_chunk_id]
                if parent_chunks:
                    parent_section_code = parent_chunks[0].metadata.section_code
                    break

        # Build code
        if parent_section_code:
            return f"{parent_section_code}.{current_section_number}"
        else:
            return current_section_number

    def _get_current_module(self) -> str:
        """Lấy module (section level 1) hiện tại"""
        for stack_type, stack_number, stack_level, _ in self.hierarchy_stack:
            if stack_type == 'section_1':
                return f"Section {stack_number}"
        return "Root"

    def _build_title_path(self, current_title: str) -> List[str]:
        """Xây dựng title path từ root đến current"""
        path = []

        # Tạo map chunk_id -> chunk để lookup nhanh
        chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}

        # Thêm titles từ stack (trừ current node vì chưa có trong chunks)
        for _, _, _, stack_chunk_id in self.hierarchy_stack[:-1]:
            if stack_chunk_id in chunk_map:
                path.append(chunk_map[stack_chunk_id].metadata.section_title)

        # Thêm current title
        path.append(current_title)

        return path

    def _update_hierarchy_stack(self, section_type: str, section_number: str,
                                section_level: int, chunk_id: str):
        """
        Cập nhật hierarchy stack khi gặp section mới

        Quy tắc dựa vào level:
        - Section level thấp hơn: pop cho đến khi tìm được parent có level < current_level
        - Item: pop cho đến khi tìm được parent là section hoặc item level thấp hơn
        """
        # Pop stack cho đến khi tìm được parent hợp lệ
        while self.hierarchy_stack:
            stack_type, _, stack_level, _ = self.hierarchy_stack[-1]

            # Giữ lại các node có level < current_level
            if stack_level < section_level:
                break
            else:
                self.hierarchy_stack.pop()

        # Thêm current section vào stack
        self.hierarchy_stack.append((section_type, section_number, section_level, chunk_id))

    def _get_parent_id(self) -> Optional[str]:
        """Lấy parent ID từ stack"""
        if len(self.hierarchy_stack) >= 2:
            # Parent là phần tử áp chót
            return self.hierarchy_stack[-2][3]
        elif len(self.hierarchy_stack) == 1:
            return None
        return None

    def _create_chunk(self, section_info: Dict, content: str, position: int) -> Chunk:
        """Tạo một chunk mới"""
        section_type = section_info['type']
        section_number = section_info['number']
        section_title = section_info.get('title', '')
        section_level = section_info['level']

        # Tạo CHUNK_ID duy nhất
        temp_id_base = f"{section_type}_{section_number}"
        chunk_id = self._generate_chunk_id(temp_id_base, content, position)

        # BƯỚC 1: Update stack
        self._update_hierarchy_stack(section_type, section_number, section_level, chunk_id)

        # BƯỚC 2: Lấy parent
        parent_id = self._get_parent_id()

        # BƯỚC 3: Build section_code
        section_code = self._build_section_code(section_number, section_type)

        # BƯỚC 4: Build title path
        title_path = self._build_title_path(section_title if section_title else section_info['full_text'])

        # Tính level dựa vào parent
        if parent_id:
            parent = [c for c in self.chunks if c.chunk_id == parent_id]
            parent_level = parent[0].metadata.level if parent else 0
            level = parent_level + 1
        else:
            level = 0

        # Tạo metadata
        metadata = SectionMetadata(
            section_code=section_code,
            section_type=section_type,
            section_number=section_number,
            section_title=section_title if section_title else section_info['full_text'],
            parent_id=parent_id,
            title_path=title_path,
            module=self._get_current_module(),
            level=level,
            position=position,
            word_count=len(content.split()),
            is_global_context=not self.found_first_section
        )

        # Tạo chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata
        )

        return chunk

    def _get_item_marker(self, chunk: Chunk) -> str:
        """Lấy ký hiệu đánh dấu của item"""
        if chunk.metadata.section_type == 'item_number':
            return chunk.metadata.section_number + '.'
        elif chunk.metadata.section_type == 'item_dash':
            return '-'
        elif chunk.metadata.section_type == 'item_plus':
            return '+'
        return ''

    def _collect_descendants_content(self, chunk: Chunk, chunk_map: dict,
                                    chunks_to_remove: list, indent: int = 0) -> str:
        """
        Đệ quy collect content của chunk và tất cả descendants

        Returns: Multi-line string với tất cả content
        """
        marker = self._get_item_marker(chunk)
        content_lines = [f"{marker} {chunk.content}"]

        # Đệ quy với children
        for child_id in chunk.metadata.children_ids:
            if child_id in chunk_map:
                child = chunk_map[child_id]
                # Thu thập content của child và descendants của nó
                child_content = self._collect_descendants_content(
                    child, chunk_map, chunks_to_remove, indent + 1
                )
                content_lines.append(child_content)
                # Đánh dấu child để xóa
                if child.chunk_id not in chunks_to_remove:
                    chunks_to_remove.append(child.chunk_id)

        return '\n'.join(content_lines)

    def _merge_short_items_to_parent(self, min_length: int = 50):
        """
        Merge items có content ngắn vào section parent

        Logic:
        - Duyệt qua tất cả chunks
        - Nếu chunk là item (item_number, item_dash, item_plus)
        - VÀ content < min_length
        - Thì append content vào parent section (bao gồm cả descendants)
        - Xóa chunk item và descendants khỏi danh sách
        """
        print(f"\n🔄 Đang merge items ngắn (< {min_length} chars) vào parent...")

        chunks_to_remove = []
        chunk_map = {c.chunk_id: c for c in self.chunks}
        merge_count = 0

        for chunk in self.chunks:
            # Bỏ qua nếu đã được đánh dấu xóa
            if chunk.chunk_id in chunks_to_remove:
                continue

            # Chỉ merge items
            if chunk.metadata.section_type not in ['item_number', 'item_dash', 'item_plus']:
                continue

            # Chỉ merge nếu content ngắn
            if len(chunk.content) >= min_length:
                continue

            # Tìm parent
            parent_id = chunk.metadata.parent_id
            if not parent_id or parent_id not in chunk_map:
                continue

            parent = chunk_map[parent_id]

            # Collect content của chunk và tất cả descendants
            full_content = self._collect_descendants_content(chunk, chunk_map, chunks_to_remove)
            parent.content += f"\n{full_content}"

            # Đánh dấu chunk này để xóa
            if chunk.chunk_id not in chunks_to_remove:
                chunks_to_remove.append(chunk.chunk_id)
            merge_count += 1

        # Xóa chunks đã merge
        self.chunks = [c for c in self.chunks if c.chunk_id not in chunks_to_remove]

        # Update children_ids của parents
        for chunk in self.chunks:
            chunk.metadata.children_ids = [
                cid for cid in chunk.metadata.children_ids
                if cid not in chunks_to_remove
            ]

        print(f"✅ Đã merge {merge_count} items (+ descendants) vào parent")
        print(f"📊 Số chunks sau merge: {len(self.chunks)}")

    def parse_document(self, docx_path: str) -> List[Chunk]:
        """Parse văn bản hướng dẫn Tendoo"""
        print(f"\n{'='*80}")
        print(f"BẮT ĐẦU PARSE TÀI LIỆU TENDOO")
        print(f"{'='*80}\n")

        doc = Document(docx_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        print(f"Tổng số đoạn văn: {len(paragraphs)}")

        # Tạo root chunk
        root_chunk_id = self._generate_chunk_id("ROOT", "Root Section", 0)
        self.hierarchy_stack.append(('root', 'ROOT', 0, root_chunk_id))

        current_section = None
        current_content = []
        position = 0

        for para in paragraphs:
            # Detect section type
            section_info = self._detect_section_type(para)

            if section_info:
                # Lưu section trước đó
                if current_section:
                    content_text = '\n'.join(current_content)
                    chunk = self._create_chunk(current_section, content_text, position)
                    self.chunks.append(chunk)
                    position += 1

                # Kiểm tra nếu đây là section đầu tiên
                if not self.found_first_section and section_info['type'].startswith('section_'):
                    self.found_first_section = True

                    # Tạo global context chunk
                    if self.global_context_content:
                        global_content = '\n'.join(self.global_context_content)
                        root_metadata = SectionMetadata(
                            section_code="ROOT",
                            section_type="root",
                            section_number="ROOT",
                            section_title="Phần mở đầu",
                            parent_id=None,
                            title_path=["Root"],
                            module="Root",
                            level=0,
                            position=0,
                            word_count=len(global_content.split()),
                            is_global_context=True
                        )

                        root_chunk = Chunk(
                            chunk_id=root_chunk_id,
                            content=global_content,
                            metadata=root_metadata
                        )

                        self.chunks.insert(0, root_chunk)
                        position += 1

                # Bắt đầu section mới
                current_section = section_info
                # Nếu là item, chỉ lấy phần text (không có ký hiệu đánh dấu)
                # Nếu là section, lấy full text
                if section_info['type'] in ['item_number', 'item_dash', 'item_plus']:
                    current_content = [section_info['title']] if section_info['title'] else []
                else:
                    current_content = [para]
            else:
                # Nội dung thuộc section hiện tại
                current_content.append(para)

                # Nếu chưa tìm thấy section đầu tiên, thêm vào global context
                if not self.found_first_section:
                    self.global_context_content.append(para)

        # Lưu section cuối cùng
        if current_section and current_content:
            content_text = '\n'.join(current_content)
            chunk = self._create_chunk(current_section, content_text, position)
            self.chunks.append(chunk)

        print(f"\n{'='*80}")
        print(f"HOÀN THÀNH PARSING")
        print(f"Tổng số chunks: {len(self.chunks)}")
        print(f"{'='*80}\n")

        # Merge items ngắn vào parent
        self._merge_short_items_to_parent(min_length=50)

        return self.chunks

    def build_hierarchy_graph(self):
        """Xây dựng đồ thị phân cấp từ chunks"""
        print("\nĐang xây dựng hierarchy graph...")

        for chunk in self.chunks:
            self.hierarchy_graph.add_node(
                chunk.chunk_id,
                chunk=chunk,
                section_code=chunk.metadata.section_code,
                section_type=chunk.metadata.section_type
            )

            if chunk.metadata.parent_id:
                self.hierarchy_graph.add_edge(
                    chunk.metadata.parent_id,
                    chunk.chunk_id,
                    relation='parent-child'
                )

        print(f"Hierarchy graph: {self.hierarchy_graph.number_of_nodes()} nodes, "
              f"{self.hierarchy_graph.number_of_edges()} edges")

    def build_sibling_relationships(self):
        """Xây dựng quan hệ anh em và related_ids"""
        print("\nĐang xây dựng sibling relationships...")

        parent_children = defaultdict(list)
        for chunk in self.chunks:
            if chunk.metadata.parent_id:
                parent_children[chunk.metadata.parent_id].append(chunk.chunk_id)

        for parent_id, children_ids in parent_children.items():
            for chunk in self.chunks:
                if chunk.chunk_id in children_ids:
                    siblings = [cid for cid in children_ids if cid != chunk.chunk_id]
                    chunk.metadata.sibling_ids = siblings
                    chunk.metadata.related_ids = siblings.copy()

        print(f"Đã xây dựng sibling relationships")

    def update_children_ids(self):
        """Cập nhật children_ids cho tất cả chunks"""
        print("\nĐang cập nhật children IDs...")

        for chunk in self.chunks:
            children = [
                c.chunk_id for c in self.chunks
                if c.metadata.parent_id == chunk.chunk_id
            ]
            chunk.metadata.children_ids = children

        print(f"Đã cập nhật children IDs")

    def extract_tags(self):
        """Tự động gán tags dựa trên content"""
        print("\nĐang trích xuất tags...")

        tag_keywords = {
            'cài đặt': ['cài đặt', 'thiết lập', 'setup', 'install'],
            'cửa hàng': ['cửa hàng', 'shop', 'store'],
            'bán hàng': ['bán hàng', 'sale', 'selling'],
            'sản phẩm': ['sản phẩm', 'product', 'item'],
            'thanh toán': ['thanh toán', 'payment', 'pay'],
            'hóa đơn': ['hóa đơn', 'invoice', 'receipt', 'bill'],
            'quy trình': ['quy trình', 'workflow', 'process'],
            'khách hàng': ['khách hàng', 'customer', 'client'],
            'thông tin': ['thông tin', 'information', 'info'],
            'mẫu': ['mẫu', 'template', 'format'],
        }

        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            tags = []

            for tag, keywords in tag_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    tags.append(tag)

            chunk.metadata.tags = tags

        print(f"Đã trích xuất tags")

    def process_document(self, docx_path: str) -> List[Chunk]:
        """Xử lý toàn bộ tài liệu"""
        self.parse_document(docx_path)
        self.build_hierarchy_graph()
        self.build_sibling_relationships()
        self.update_children_ids()
        self.extract_tags()
        return self.chunks

    def save_chunks(self, output_path: str):
        """Lưu chunks ra JSON"""
        print(f"\nĐang lưu chunks vào {output_path}...")

        chunks_data = [chunk.to_dict() for chunk in self.chunks]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        print(f"Đã lưu {len(chunks_data)} chunks")

    def save_graph(self, output_dir: str):
        """Lưu graph data"""
        os.makedirs(output_dir, exist_ok=True)

        graph_copy = self.hierarchy_graph.copy()
        for node in graph_copy.nodes():
            if 'chunk' in graph_copy.nodes[node]:
                del graph_copy.nodes[node]['chunk']

        graph_path = os.path.join(output_dir, 'hierarchy_graph.gexf')
        nx.write_gexf(graph_copy, graph_path)
        print(f"Đã lưu hierarchy graph: {graph_path}")

    def print_summary(self):
        """In tóm tắt kết quả"""
        print("\n" + "="*80)
        print("TỔNG KẾT CHUNKING")
        print("="*80)

        print(f"\n📊 Thống kê cơ bản:")
        print(f"  - Tổng số chunks: {len(self.chunks)}")
        print(f"  - Hierarchy edges: {self.hierarchy_graph.number_of_edges()}")

        type_counts = defaultdict(int)
        for chunk in self.chunks:
            type_counts[chunk.metadata.section_type] += 1

        print(f"\n📁 Phân bố theo section type:")
        for stype in sorted(type_counts.keys()):
            print(f"  - {stype}: {type_counts[stype]} chunks")

        global_chunks = [c for c in self.chunks if c.metadata.is_global_context]
        print(f"\n🌍 Global context chunks: {len(global_chunks)}")

        print("\n" + "="*80 + "\n")

    def print_sample_chunks(self, n: int = 5):
        """In các chunk mẫu"""
        print("\n" + "="*80)
        print(f"MẪU {n} CHUNKS ĐẦU TIÊN")
        print("="*80 + "\n")

        for i, chunk in enumerate(self.chunks[:n]):
            print(f"\n{'─'*80}")
            print(f"Chunk #{i+1}")
            print(f"{'─'*80}")
            print(f"ID: {chunk.chunk_id}")
            print(f"Section Code: {chunk.metadata.section_code}")
            print(f"Section Type: {chunk.metadata.section_type}")
            print(f"Section Title: {chunk.metadata.section_title}")
            print(f"Title Path: {' > '.join(chunk.metadata.title_path)}")
            print(f"Module: {chunk.metadata.module}")
            print(f"Level: {chunk.metadata.level}")
            print(f"Parent ID: {chunk.metadata.parent_id}")
            print(f"Children IDs: {chunk.metadata.children_ids[:3]}..." if len(chunk.metadata.children_ids) > 3 else f"Children IDs: {chunk.metadata.children_ids}")
            print(f"Sibling IDs: {len(chunk.metadata.sibling_ids)} siblings")
            print(f"Tags: {chunk.metadata.tags}")
            print(f"Is Global Context: {chunk.metadata.is_global_context}")
            print(f"Word Count: {chunk.metadata.word_count}")
            print(f"\nContent Preview:")
            preview = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
            print(preview)

        print("\n" + "="*80 + "\n")


def main():
    """Hàm main"""

    # Cấu hình
    DOCX_PATH = "TÀI LIỆU HƯỚNG DẪN SỬ DỤNG TENDOO APP (3).docx"  # Thay đổi tên file của bạn
    OUTPUT_DIR = "output_tendoo"

    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Khởi tạo chunker
    chunker = TendooDocumentChunker()

    # Xử lý tài liệu
    chunks = chunker.process_document(DOCX_PATH)

    # Lưu kết quả
    chunker.save_chunks(os.path.join(OUTPUT_DIR, 'chunks.json'))
    chunker.save_graph(OUTPUT_DIR)

    # In tóm tắt
    chunker.print_summary()

    # In mẫu chunks
    chunker.print_sample_chunks(10)


if __name__ == "__main__":
    main()
