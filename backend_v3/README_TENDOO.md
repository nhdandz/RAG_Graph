# Tendoo RAG System

Hệ thống RAG (Retrieval-Augmented Generation) cho tài liệu hướng dẫn sử dụng Tendoo App.

## Cấu trúc tài liệu Tendoo

Hệ thống hỗ trợ cấu trúc phân cấp sau:

```
TÀI LIỆU HƯỚNG DẪN
1. Mục chính
  1.1. Mục con cấp 1
    1.1.1. Mục con cấp 2
      1.1.1.1. Mục con cấp 3
        1 Các bước đánh số
        2 Trong mục
        - Các ý gạch ngang
        - Trong mục
        + Các ý dấu cộng
        + Trong mục
```

## Cài đặt

### 1. Cài đặt các thư viện cần thiết

```bash
pip install fastapi uvicorn python-docx networkx scikit-learn google-generativeai
```

### 2. Cài đặt Ollama (cho embedding)

Tải và cài đặt Ollama từ: https://ollama.ai/

Sau đó tải model BGE-M3:

```bash
ollama pull bge-m3
```

## Sử dụng

### Bước 1: Tạo file tài liệu mẫu

```bash
cd backend_v3
python create_tendoo_sample.py
```

Lệnh này sẽ tạo file `tendoo_guide.docx` với cấu trúc mẫu.

### Bước 2: Test chunking

```bash
python test_tendoo.py
```

Lệnh này sẽ:
- Parse file docx
- Tạo chunks theo cấu trúc phân cấp
- Xây dựng đồ thị quan hệ
- Lưu kết quả vào thư mục `output_tendoo/`
- In ra thống kê và mẫu chunks

### Bước 3: Chạy RAG API

```bash
python tendoo_rag.py
```

Hoặc sử dụng uvicorn:

```bash
uvicorn tendoo_rag:app --reload --port 8001
```

API sẽ chạy tại: http://localhost:8001

### Bước 4: Test query

Mở trình duyệt hoặc sử dụng curl để test:

**Health check:**
```bash
curl http://localhost:8001/
```

**Query:**
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Làm thế nào để cài đặt thông tin cửa hàng?",
    "topK": 5
  }'
```

**Xem thống kê:**
```bash
curl http://localhost:8001/stats
```

## Cấu trúc file

```
backend_v3/
├── tendoo_rag_chunking.py      # Logic chunking chính
├── create_tendoo_sample.py     # Tạo file docx mẫu
├── test_tendoo.py              # Script test chunking
├── tendoo_rag.py               # RAG API
├── tendoo_guide.docx           # File tài liệu mẫu (tự tạo)
├── output_tendoo/              # Kết quả chunking
│   ├── chunks.json             # Chunks đã tạo
│   └── hierarchy_graph.gexf    # Đồ thị phân cấp
└── README_TENDOO.md            # File này
```

## Sử dụng với tài liệu riêng

### 1. Chuẩn bị file docx

Tạo file Word (.docx) với cấu trúc phân cấp theo format:

- `1. Tiêu đề mục chính`
- `1.1. Tiêu đề mục con`
- `1.1.1. Tiêu đề mục con cấp 2`
- `1.2.1.1. Tiêu đề mục con cấp 3` (nếu cần)

Bên trong mỗi mục có thể có:
- Các bước đánh số: `1 Nội dung`, `2 Nội dung`
- Các ý gạch ngang: `- Nội dung`
- Các ý dấu cộng: `+ Nội dung`

### 2. Cập nhật đường dẫn

Trong file `tendoo_rag_chunking.py`, sửa dòng:

```python
DOCX_PATH = "tendoo_guide.docx"  # Thay bằng tên file của bạn
```

Hoặc truyền trực tiếp khi gọi:

```python
chunker = TendooDocumentChunker()
chunks = chunker.process_document("path/to/your/file.docx")
```

### 3. Chạy chunking

```bash
python test_tendoo.py
```

### 4. Khởi động API

```bash
python tendoo_rag.py
```

## Ví dụ queries

1. **Hỏi về thông tin cụ thể:**
   ```
   "Làm thế nào để cập nhật thông tin cửa hàng?"
   "Tendoo hỗ trợ những phương thức thanh toán nào?"
   ```

2. **Hỏi về quy trình:**
   ```
   "Quy trình bán hàng cho shop FnB như thế nào?"
   "Các bước để nhập hàng vào kho là gì?"
   ```

3. **Hỏi về danh sách:**
   ```
   "Có những mẫu hóa đơn nào?"
   "Thông tin sản phẩm cần có những gì?"
   ```

4. **Hỏi chung:**
   ```
   "Hướng dẫn tối ưu bán hàng"
   "Cách quản lý kho hàng"
   ```

## Tùy chỉnh

### Thay đổi số lượng chunks trả về

Trong `tendoo_rag.py`:

```python
class Config:
    TOP_K = 5  # Số chunks tương đồng nhất
    MAX_DESCENDANTS = 5  # Số con tối đa
    MAX_SIBLINGS = 3  # Số anh em tối đa
    INCLUDE_PARENT = True  # Có bao gồm chunk cha không
```

### Thay đổi model

```python
class Config:
    LLM_MODEL = "gemini-2.5-flash"  # Model sinh câu trả lời
    EMBEDDING_MODEL = "bge-m3"  # Model tạo embedding
```

### Thay đổi tags

Trong `tendoo_rag_chunking.py`, method `extract_tags()`:

```python
tag_keywords = {
    'cài đặt': ['cài đặt', 'thiết lập', 'setup'],
    'bán hàng': ['bán hàng', 'sale'],
    # Thêm tags của bạn...
}
```

## API Endpoints

### GET /
Health check

**Response:**
```json
{
  "status": "ok",
  "service": "Tendoo RAG API",
  "chunks_loaded": 50,
  "embeddings_created": 50
}
```

### POST /query
Thực hiện query

**Request:**
```json
{
  "query": "Làm thế nào để cài đặt cửa hàng?",
  "topK": 5
}
```

**Response:**
```json
{
  "answer": "Để cài đặt cửa hàng trong Tendoo App...",
  "retrievedDocuments": [
    {
      "chunk_id": "abc123",
      "section_code": "1.1.1",
      "section_title": "Thông tin cửa hàng",
      "content": "...",
      "similarity": 0.85,
      "title_path": "Cài đặt cửa hàng > Cửa hàng > Thông tin cửa hàng"
    }
  ],
  "metadata": {
    "total_retrieved": 5,
    "total_enriched": 15,
    "top_k": 5
  }
}
```

### GET /stats
Xem thống kê

**Response:**
```json
{
  "total_chunks": 50,
  "by_section_type": {
    "section_1": 2,
    "section_2": 8,
    "section_3": 15
  },
  "by_level": {
    "0": 1,
    "1": 2,
    "2": 8
  },
  "top_tags": {
    "cài đặt": 10,
    "bán hàng": 8
  }
}
```

## Troubleshooting

### Lỗi: "File chunks.json không tồn tại"

Hãy chạy `python test_tendoo.py` trước để tạo chunks.

### Lỗi: "Cannot connect to Ollama"

Kiểm tra Ollama đã chạy chưa:
```bash
ollama list
```

Nếu chưa, khởi động Ollama và pull model:
```bash
ollama pull bge-m3
```

### Lỗi: "Gemini API key invalid"

Cập nhật API key trong `tendoo_rag.py`:
```python
class Config:
    GEMINI_API_KEY = "your-api-key-here"
```

## License

MIT
