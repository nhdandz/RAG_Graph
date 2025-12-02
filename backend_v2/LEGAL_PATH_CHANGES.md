# Legal Path Enhancement - Summary of Changes

## Problem
Response hiện tại chỉ có `title_path` (text hierarchy) mà không có thông tin về cấu trúc pháp luật (Chương/Điều/Khoản), khiến người dùng không biết đang xem nội dung thuộc điều khoản nào.

**Ví dụ lỗi trước đây:**
```
title_path: "Phần mở đầu > TUYỂN SINH > Tổ chức khám sức khỏe > 3. Hằng năm..."
```
→ Không biết đây là Chương mấy, Điều mấy, Khoản mấy!

## Solution
Thêm `legal_path` để hiển thị cấu trúc pháp luật rõ ràng:

**Sau khi sửa:**
```
legal_path: "Chương II > Điều 6 > Khoản 3"
title_path: "Phần mở đầu > TUYỂN SINH > Tổ chức khám sức khỏe > 3. Hằng năm..."
```

## Changes Made

### 1. **Added Helper Functions** (Lines 485-578)
- `format_legal_path()`: Format single level (Chương/Điều/Khoản)
- `build_legal_hierarchy_path()`: Build full hierarchy path by traversing parents

```python
# Example output:
"Chương I > Điều 3 > Khoản 4"
```

### 2. **Enhanced Context Building** (Line 717-723)
Updated `build_enriched_context()` to include legal path:

```python
legal_path = build_legal_hierarchy_path(chunk)

enriched_parts.append("【 NỘI DUNG CHÍNH 】")
enriched_parts.append(f"📌 Tiêu đề: {section_title}")
if legal_path:
    enriched_parts.append(f"📜 Cấu trúc: {legal_path}")  # NEW!
enriched_parts.append(f"📍 Vị trí: {title_path}")
```

### 3. **Enhanced Response Structure** (Lines 1244-1255)
Added legal path to `retrievedDocuments`:

```python
retrieved_docs.append({
    "filename": "Thông tư tuyển sinh",
    "content": chunk['content'],
    "similarity": chunk['similarity'],
    "section_code": metadata.get('section_code', ''),
    "section_title": metadata.get('section_title', ''),
    "section_type": metadata.get('section_type', ''),      # NEW!
    "legal_path": legal_path,                               # NEW!
    "title_path": ' > '.join(metadata.get('title_path', []))
})
```

### 4. **Enhanced Context Structure** (Lines 778-784)
Updated `build_multi_chunk_context()` to use legal path in `headingPath`:

```python
legal_path = build_legal_hierarchy_path(chunk)
display_path = legal_path if legal_path else title_path

context_chunks.append(ContextChunk(
    headingPath=display_path,  # Now uses legal path!
    # ...
))
```

### 5. **Updated LLM Prompt** (Lines 943-962)
Enhanced prompt to instruct LLM to cite using legal structure:

```
TRÍCH DẪN: Dựa vào "Cấu trúc pháp luật" để trích dẫn chính xác
  Ví dụ: "Theo Chương II, Điều 6, Khoản 1..."
  Ví dụ: "Căn cứ Điều 3, Khoản 4..."
```

## Response Format Changes

### Before:
```json
{
  "retrievedDocuments": [
    {
      "section_code": "I.3.4",
      "section_title": "4. Tuyển sinh đủ số lượng...",
      "similarity": 0.85
    }
  ]
}
```

### After:
```json
{
  "retrievedDocuments": [
    {
      "section_code": "I.3.4",
      "section_title": "4. Tuyển sinh đủ số lượng...",
      "section_type": "khoan",
      "legal_path": "Chương I > Điều 3 > Khoản 4",
      "title_path": "Phần mở đầu > NHỮNG QUY ĐỊNH CHUNG > Nguyên tắc tuyển sinh > ...",
      "similarity": 0.85
    }
  ]
}
```

## Testing

Run test script to verify:
```bash
python test_legal_path.py
```

Expected output:
```
✓ Legal Path : Chương I > Điều 3 > Khoản 4
```

## Benefits

1. **Clear Legal Citation**: Users can now see exact legal structure (Chapter/Article/Clause)
2. **Better UX**: Frontend can display both legal path and descriptive path
3. **Accurate References**: LLM can cite specific articles/clauses correctly
4. **Compliance**: Matches legal document citation standards

## Frontend Integration

Frontend should display both paths:

```tsx
<div>
  <div className="legal-path">
    📜 {doc.legal_path}
  </div>
  <div className="title-path">
    📍 {doc.title_path}
  </div>
</div>
```

Example display:
```
📜 Chương II > Điều 6 > Khoản 3
📍 Phần mở đầu > TUYỂN SINH ĐÀO TẠO > Tổ chức khám sức khỏe > 3. Hằng năm...
```

## Backward Compatibility

✅ All existing fields remain unchanged
✅ Only added new fields (`legal_path`, `section_type`)
✅ No breaking changes
