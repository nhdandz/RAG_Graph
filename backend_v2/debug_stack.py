#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug stack issue bằng cách re-run với logging"""

import os
import re
from docx import Document

# Tạo một version mini với logging
class StackDebugger:
    def __init__(self):
        self.stack = []
        self.chunks_created = []

    def update_stack(self, section_type, section_number):
        """Update stack và log"""
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

        # Pop stack
        while self.stack:
            stack_type, stack_number = self.stack[-1]
            stack_level = hierarchy_order[stack_type]

            if stack_level < current_level:
                break
            else:
                popped = self.stack.pop()
               # print(f"    POP: {popped}")

        # Push
        self.stack.append((section_type, section_number))
        #print(f"    PUSH: ({section_type}, {section_number})")

    def build_section_code(self):
        """Build section code từ stack"""
        code_parts = []
        for stack_type, stack_number in self.stack:
            if stack_type != 'root':
                code_parts.append(stack_number)
        return '.'.join(code_parts)

    def detect_section(self, text):
        """Detect section type"""
        text = text.strip()

        # Chương
        match = re.match(r'^Chương\s+([IVX]+)\.\s*(.+)$', text, re.IGNORECASE)
        if match:
            return {'type': 'chuong', 'number': match.group(1), 'title': match.group(2).strip()}

        # Mục
        match = re.match(r'^Mục\s+(\d+)\.\s*(.+)$', text, re.IGNORECASE)
        if match:
            return {'type': 'muc', 'number': match.group(1), 'title': match.group(2).strip()}

        # Điều
        match = re.match(r'^Điều\s+(\d+)\.\s*(.+)$', text, re.IGNORECASE)
        if match:
            return {'type': 'dieu', 'number': match.group(1), 'title': match.group(2).strip()}

        # Khoản
        match = re.match(r'^(\d+)\.\s+(.+)$', text)
        if match:
            return {'type': 'khoan', 'number': match.group(1), 'title': ''}

        return None

    def process_document(self, docx_path):
        """Process document"""
        doc = Document(docx_path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        # Initialize stack với ROOT
        self.stack.append(('root', 'ROOT'))

        for i, para in enumerate(paragraphs):
            section_info = self.detect_section(para)

            if section_info:
                stype = section_info['type']
                snum = section_info['number']
                stitle = section_info['title']

                # Chỉ log Chương III, Mục 1 của Chương VIII, và Điều 8
                if (stype == 'chuong' and snum == 'III') or \
                   (stype == 'chuong' and snum == 'VIII') or \
                   (stype == 'muc' and snum == '1') or \
                   (stype == 'dieu' and snum == '8'):

                    print(f"\n{'='*80}")
                    print(f"Đang xử lý: {stype} {snum} - {stitle}")
                    print(f"Stack TRƯỚC khi update: {self.stack}")

                    # Update stack
                    self.update_stack(stype, snum)

                    # Build section code
                    section_code = self.build_section_code()

                    print(f"Stack SAU khi update: {self.stack}")
                    print(f"Section code: {section_code}")

                    self.chunks_created.append({
                        'code': section_code,
                        'type': stype,
                        'number': snum,
                        'title': stitle
                    })
                else:
                    # Silent update
                    self.update_stack(stype, snum)

# Run
debugger = StackDebugger()
debugger.process_document('Source.docx')

print(f"\n\n{'='*80}")
print("TÓM TẮT CÁC CHUNK QUAN TRỌNG")
print(f"{'='*80}\n")
for chunk in debugger.chunks_created:
    print(f"{chunk['code']:20s} | {chunk['type']:8s} {chunk['number']:5s} | {chunk['title']}")
