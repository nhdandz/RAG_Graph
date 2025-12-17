# -*- coding: utf-8 -*-
"""
Hybrid Hierarchical-Graph Chunking System for Admission Documents
Hệ thống phân tích văn bản tuyển sinh theo cấu trúc phân cấp và đồ thị quan hệ
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
    Metadata cho mỗi chunk theo cấu trúc Thông tư tuyển sinh

    Layer 1: Hierarchical structure (section_code)
    Layer 2: Graph relationships (parent/children/siblings/related)
    Layer 3: Metadata (tags, module, titlePath)
    """
    # Layer 1: Hierarchical Structure
    section_code: str  # VD: "I", "I.1", "I.1.1", "I.1.1.a", "I.1.1.a.-", "I.1.1.a.-+"
    section_type: str  # "root" | "chuong" | "muc" | "dieu" | "khoản" | "item_abc" | "item_dash" | "item_plus"
    section_number: str  # Số thứ tự: "I", "1", "a", "-", "+"
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
    is_global_context: bool = False  # Phần đầu đến Chương I là global context


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


class AdmissionDocumentChunker:
    """
    Parser chuyên biệt cho văn bản tuyển sinh

    Cấu trúc phân cấp:
    - Root (Phần đầu đến Chương I - kiến thức chung)
    - Chương (I, II, III, ...)
    - Mục (1, 2, 3, ...) - có thể không có, nhảy thẳng sang Điều
    - Điều (1, 2, 3, ...)
    - Khoản (1, 2, 3, ...)
    - Các ý a, b, c
    - Các ý gạch ngang -
    - Các ý dấu cộng +
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.hierarchy_graph = nx.DiGraph()  # Đồ thị phân cấp
        self.semantic_graph = nx.Graph()  # Đồ thị ngữ nghĩa

        # Stack để track hierarchy hiện tại
        self.hierarchy_stack: List[Tuple[str, str, int]] = []  # (type, code, chunk_id)

        # Global context tracking
        self.global_context_content: List[str] = []
        self.found_chapter_one = False

    def _generate_chunk_id(self, section_code: str, text: str, position: int = 0) -> str:
        """Tạo ID duy nhất cho chunk

        Args:
            section_code: Mã section (có thể là tạm thời)
            text: Nội dung
            position: Vị trí trong văn bản để đảm bảo unique
        """
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
                'full_text': str
            }
        """
        text = text.strip()

        # Pattern cho Chương (Chương I, Chương II, ...)
        chuong_pattern = r'^Chương\s+([IVX]+)\.\s*(.+)$'
        match = re.match(chuong_pattern, text, re.IGNORECASE)
        if match:
            return {
                'type': 'chuong',
                'number': match.group(1),
                'title': match.group(2).strip(),
                'full_text': text
            }

        # Pattern cho Mục (Mục 1., Mục 2., ...)
        muc_pattern = r'^Mục\s+(\d+)\.\s*(.+)$'
        match = re.match(muc_pattern, text, re.IGNORECASE)
        if match:
            return {
                'type': 'muc',
                'number': match.group(1),
                'title': match.group(2).strip(),
                'full_text': text
            }

        # Pattern cho Điều (Điều 1., Điều 2., ...)
        dieu_pattern = r'^Điều\s+(\d+)\.\s*(.+)$'
        match = re.match(dieu_pattern, text, re.IGNORECASE)
        if match:
            return {
                'type': 'dieu',
                'number': match.group(1),
                'title': match.group(2).strip(),
                'full_text': text
            }

        # Pattern cho Khoản (1., 2., 3., ...)
        khoan_pattern = r'^(\d+)\.\s+(.+)$'
        match = re.match(khoan_pattern, text)
        if match:
            return {
                'type': 'khoan',
                'number': match.group(1),
                'title': '',
                'full_text': text
            }

        # Pattern cho các ý a), b), c), đ) (bao gồm cả chữ Việt và in hoa)
        abc_pattern = r'^([a-zđ])\)\s+(.+)$'
        match = re.match(abc_pattern, text, re.IGNORECASE)
        if match:
            return {
                'type': 'item_abc',
                'number': match.group(1).lower(),  # Chuẩn hóa về chữ thường
                'title': '',
                'full_text': text
            }

        # Pattern cho các ý gạch ngang -
        dash_pattern = r'^-\s+(.+)$'
        match = re.match(dash_pattern, text)
        if match:
            return {
                'type': 'item_dash',
                'number': '-',
                'title': '',
                'full_text': text
            }

        # Pattern cho các ý dấu cộng +
        plus_pattern = r'^\+\s+(.+)$'
        match = re.match(plus_pattern, text)
        if match:
            return {
                'type': 'item_plus',
                'number': '+',
                'title': '',
                'full_text': text
            }

        return None

    def _build_section_code(self) -> str:
        """
        Xây dựng section_code từ hierarchy stack

        VD: "I.1.3.a.-"

        Lưu ý: Hàm này được gọi SAU KHI current node đã được thêm vào stack
        Do đó stack[-1] chính là current node, ta chỉ cần duyệt toàn bộ stack
        """
        # Lấy tất cả codes từ stack (bao gồm cả current node)
        code_parts = []

        for stack_type, stack_number, _ in self.hierarchy_stack:
            # Bỏ qua ROOT trong section code
            if stack_type != 'root':
                code_parts.append(stack_number)

        return '.'.join(code_parts)

    def _get_current_module(self) -> str:
        """Lấy module (Chương) hiện tại"""
        for stack_type, stack_number, _ in self.hierarchy_stack:
            if stack_type == 'chuong':
                return f"Chương {stack_number}"
        return "Root"

    def _build_title_path(self, current_title: str) -> List[str]:
        """
        Xây dựng title path từ root đến current

        Lưu ý: Hàm này được gọi SAU KHI current node đã được thêm vào stack
        Current node chưa có chunk trong self.chunks, nên cần thêm riêng current_title
        """
        path = []

        # Tạo map chunk_id -> chunk để lookup nhanh
        chunk_map = {chunk.chunk_id: chunk for chunk in self.chunks}

        # Thêm titles từ stack (trừ current node vì chưa có trong chunks)
        for _, _, stack_chunk_id in self.hierarchy_stack[:-1]:  # Bỏ qua phần tử cuối (current)
            if stack_chunk_id in chunk_map:
                path.append(chunk_map[stack_chunk_id].metadata.section_title)

        # Thêm current title
        path.append(current_title)

        return path

    def _update_hierarchy_stack(self, section_type: str, section_number: str, chunk_id: str):
        """
        Cập nhật hierarchy stack khi gặp section mới

        Quy tắc:
        - Chương: clear stack, chỉ giữ root
        - Mục: pop cho đến Chương
        - Điều: pop cho đến Mục hoặc Chương (phụ thuộc vào có Mục hay không)
        - Khoản: pop cho đến Điều
        - item_abc: pop cho đến Khoản
        - item_dash: pop cho đến item_abc
        - item_plus: pop cho đến item_dash

        Lưu ý: Khi gặp section cùng level (ví dụ Điều 3 sau Điều 2),
        cần pop section cùng level đó ra để lấy parent đúng
        """
        hierarchy_order = {
            'root': 0,
            'chuong': 1,
            'muc': 2,
            'dieu': 3,
            'khoan': 4,
            'item_abc': 5,
            'item_dash': 6,
            'item_plus': 7
        }

        current_level = hierarchy_order[section_type]

        # Pop stack cho đến khi tìm được parent hợp lệ
        # Parent hợp lệ là node có level NHỎ HƠN THỰC SỰ (strictly less than)
        while self.hierarchy_stack:
            stack_type, _, _ = self.hierarchy_stack[-1]
            stack_level = hierarchy_order[stack_type]

            # Chỉ giữ lại các node có level < current_level
            # Pop tất cả các node có level >= current_level
            if stack_level < current_level:
                break
            else:
                self.hierarchy_stack.pop()

        # Thêm current section vào stack
        self.hierarchy_stack.append((section_type, section_number, chunk_id))

    def _get_parent_id(self) -> Optional[str]:
        """
        Lấy parent ID từ stack

        Lưu ý: Hàm này được gọi SAU KHI _update_hierarchy_stack đã thêm current node vào stack
        Do đó:
        - stack[-1] là current node
        - stack[-2] là parent node (nếu có)
        """
        if len(self.hierarchy_stack) >= 2:
            # Parent là phần tử áp chót
            return self.hierarchy_stack[-2][2]
        elif len(self.hierarchy_stack) == 1:
            # Nếu chỉ có 1 phần tử, có nghĩa là nó là root, không có parent
            return None
        return None

    def _create_chunk(self, section_info: Dict, content: str, position: int) -> Chunk:
        """Tạo một chunk mới"""
        section_type = section_info['type']
        section_number = section_info['number']
        section_title = section_info.get('title', '')

        # TẠO CHUNK_ID duy nhất dựa trên position
        # Sử dụng position để đảm bảo không bị trùng ID giữa các sections cùng type
        temp_id_base = f"{section_type}_{section_number}"
        chunk_id = self._generate_chunk_id(temp_id_base, content, position)

        # BƯỚC 1: Update stack TRƯỚC để tìm parent đúng
        # Sau khi update, stack chỉ chứa các ancestor thực sự + current node
        self._update_hierarchy_stack(section_type, section_number, chunk_id)

        # BƯỚC 2: Lấy parent SAU KHI đã update stack
        # Bây giờ parent_id sẽ là phần tử áp chót của stack
        parent_id = self._get_parent_id()

        # BƯỚC 3: Build section_code SAU KHI update stack
        section_code = self._build_section_code()

        # BƯỚC 4: Build title path SAU KHI update stack
        title_path = self._build_title_path(section_title if section_title else section_info['full_text'])

        # Tính level dựa vào parent
        if parent_id:
            parent = [c for c in self.chunks if c.chunk_id == parent_id]
            parent_level = parent[0].metadata.level if parent else 0
            level = parent_level + 1
        else:
            # Không có parent => đây là root
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
            is_global_context=not self.found_chapter_one
        )

        # Tạo chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata
        )

        return chunk

    def parse_document(self, docx_path: str) -> List[Chunk]:
        """
        Parse văn bản tuyển sinh

        Chiến lược:
        1. Đọc toàn bộ paragraphs
        2. Phát hiện sections theo patterns
        3. Nhóm content cho mỗi section
        4. Tạo chunks với metadata đầy đủ
        """
        print(f"\n{'='*80}")
        print(f"BẮT ĐẦU PARSE TÀI LIỆU TUYỂN SINH")
        print(f"{'='*80}\n")

        doc = Document(docx_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        print(f"Tổng số đoạn văn: {len(paragraphs)}")

        # Tạo root chunk cho phần đầu (global context)
        root_chunk_id = self._generate_chunk_id("ROOT", "Root Section", 0)
        self.hierarchy_stack.append(('root', 'ROOT', root_chunk_id))

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

                # Kiểm tra nếu đây là Chương I
                if section_info['type'] == 'chuong' and section_info['number'] == 'I':
                    self.found_chapter_one = True

                    # Tạo global context chunk từ nội dung đã thu thập
                    if self.global_context_content:
                        global_content = '\n'.join(self.global_context_content)
                        root_metadata = SectionMetadata(
                            section_code="ROOT",
                            section_type="root",
                            section_number="ROOT",
                            section_title="Phần mở đầu và quy định chung",
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

                        # Insert ở đầu
                        self.chunks.insert(0, root_chunk)
                        position += 1

                # Bắt đầu section mới
                current_section = section_info
                current_content = [para]
            else:
                # Nội dung thuộc section hiện tại
                current_content.append(para)

                # Nếu chưa tìm thấy Chương I, thêm vào global context
                if not self.found_chapter_one:
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

        return self.chunks

    def build_hierarchy_graph(self):
        """Xây dựng đồ thị phân cấp từ chunks"""
        print("\nĐang xây dựng hierarchy graph...")

        for chunk in self.chunks:
            # Thêm node
            self.hierarchy_graph.add_node(
                chunk.chunk_id,
                chunk=chunk,
                section_code=chunk.metadata.section_code,
                section_type=chunk.metadata.section_type
            )

            # Thêm edge parent-child
            if chunk.metadata.parent_id:
                self.hierarchy_graph.add_edge(
                    chunk.metadata.parent_id,
                    chunk.chunk_id,
                    relation='parent-child'
                )

        print(f"Hierarchy graph: {self.hierarchy_graph.number_of_nodes()} nodes, {self.hierarchy_graph.number_of_edges()} edges")

    def build_sibling_relationships(self):
        """Xây dựng quan hệ anh em (cùng parent) và related_ids"""
        print("\nĐang xây dựng sibling relationships và related_ids...")

        # Nhóm chunks theo parent
        parent_children = defaultdict(list)
        for chunk in self.chunks:
            if chunk.metadata.parent_id:
                parent_children[chunk.metadata.parent_id].append(chunk.chunk_id)

        # Cập nhật siblings và related_ids
        for parent_id, children_ids in parent_children.items():
            for chunk in self.chunks:
                if chunk.chunk_id in children_ids:
                    # Lấy tất cả siblings (trừ chính nó)
                    siblings = [cid for cid in children_ids if cid != chunk.chunk_id]
                    chunk.metadata.sibling_ids = siblings

                    # related_ids = các chunk có cùng parent (siblings)
                    chunk.metadata.related_ids = siblings.copy()

        print(f"Đã xây dựng sibling relationships và related_ids")

    def update_children_ids(self):
        """Cập nhật children_ids cho tất cả chunks"""
        print("\nĐang cập nhật children IDs...")

        for chunk in self.chunks:
            # Tìm tất cả children
            children = [
                c.chunk_id for c in self.chunks
                if c.metadata.parent_id == chunk.chunk_id
            ]
            chunk.metadata.children_ids = children

        print(f"Đã cập nhật children IDs")

    def extract_tags(self):
        """Tự động gán tags dựa trên content"""
        print("\nĐang trích xuất tags...")

        # Các từ khóa quan trọng
        tag_keywords = {
            'tuyển sinh': ['tuyển sinh', 'thi tuyển', 'xét tuyển'],
            'hồ sơ': ['hồ sơ', 'giấy tờ', 'chứng chỉ'],
            'điều kiện': ['điều kiện', 'tiêu chuẩn', 'yêu cầu'],
            'đào tạo': ['đào tạo', 'học tập', 'chương trình'],
            'quân đội': ['quân đội', 'quân sự', 'sĩ quan'],
            'thời gian': ['thời gian', 'thời hạn', 'hạn chót'],
            'kết quả': ['kết quả', 'điểm số', 'trúng tuyển'],
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
        """
        Xử lý toàn bộ tài liệu với đầy đủ các bước
        """
        # Bước 1: Parse document
        self.parse_document(docx_path)

        # Bước 2: Build hierarchy graph
        self.build_hierarchy_graph()

        # Bước 3: Build sibling relationships
        self.build_sibling_relationships()

        # Bước 4: Update children IDs
        self.update_children_ids()

        # Bước 5: Extract tags
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

        # Tạo copy không có chunk objects
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

        # Thống kê theo type
        type_counts = defaultdict(int)
        for chunk in self.chunks:
            type_counts[chunk.metadata.section_type] += 1

        print(f"\n📁 Phân bố theo section type:")
        for stype in sorted(type_counts.keys()):
            print(f"  - {stype}: {type_counts[stype]} chunks")

        # Global context
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
            print(f"Children IDs: {chunk.metadata.children_ids}")
            print(f"Sibling IDs: {chunk.metadata.sibling_ids}")
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
    DOCX_PATH = r"Source.docx"
    OUTPUT_DIR = "output_admission"

    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Khởi tạo chunker
    chunker = AdmissionDocumentChunker()

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
