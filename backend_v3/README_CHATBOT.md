# Tendoo Customer Support Chatbot

Chatbot hỗ trợ khách hàng về Tendoo App sử dụng RAG (Retrieval-Augmented Generation).

## Tính năng

### 🎯 Tính năng chính
- **Hỏi đáp thông minh**: Trả lời câu hỏi về Tendoo App
- **Hybrid Search**: Kết hợp Dense (embedding) + Sparse (BM25)
- **Graph Enrichment**: Tự động bổ sung context từ parent/children/siblings
- **LLM Reranking**: Sắp xếp lại chunks theo mức độ liên quan
- **Multilingual**: Hỗ trợ tiếng Việt tốt

### 🔧 Công nghệ
- **LLM**: Gemini 2.0 Flash (nhanh, miễn phí)
- **Embedding**: BGE-M3 (multilingual, qua Ollama)
- **Framework**: FastAPI
- **RAG**: Advanced với nhiều tính năng tối ưu

## Cài đặt

### Yêu cầu
```bash
# Python packages
pip install fastapi uvicorn google-generativeai scikit-learn numpy

# Ollama (cho embeddings)
# Tải từ: https://ollama.ai/
ollama pull bge-m3
```

### Chuẩn bị dữ liệu
Trước tiên, cần tạo chunks từ tài liệu:
```bash
cd backend_v3

# Tạo file mẫu
/home/admin123/miniconda3/envs/py310/bin/python create_tendoo_sample.py

# Chạy chunking
/home/admin123/miniconda3/envs/py310/bin/python test_tendoo.py
```

Kết quả sẽ được lưu tại: `output_tendoo/chunks.json`

## Sử dụng

### Bước 1: Khởi động Chatbot Server

```bash
# Cách 1: Chạy trực tiếp
/home/admin123/miniconda3/envs/py310/bin/python tendoo_chatbot.py

# Cách 2: Dùng uvicorn
uvicorn tendoo_chatbot:app --reload --port 8002
```

Server sẽ chạy tại: **http://localhost:8002**

Output khi khởi động:
```
================================================================================
KHỞI ĐỘNG TENDOO CUSTOMER SUPPORT CHATBOT
================================================================================

✅ Đã load 105 chunks từ output_tendoo/chunks.json

🔄 Đang tạo embeddings cho chunks...
  Đã embed 10/105 chunks
  Đã embed 20/105 chunks
  ...
✅ Đã tạo 105 embeddings

✅ Chatbot sẵn sàng phục vụ!
📊 105 chunks đã được load
🤖 Model: gemini-2.0-flash-exp
🔍 Embedding: bge-m3
================================================================================
```

### Bước 2: Test Chatbot

Mở terminal mới và chạy:
```bash
/home/admin123/miniconda3/envs/py310/bin/python test_chatbot.py
```

Script sẽ:
1. Kiểm tra server đang chạy
2. Test với 5 câu hỏi mẫu
3. Chuyển sang chế độ tương tác

## API Endpoints

### 1. Health Check
```bash
GET http://localhost:8002/

Response:
{
  "status": "ok",
  "service": "Tendoo Customer Support Chatbot",
  "version": "1.0",
  "chunks_loaded": 105,
  "embeddings_created": 105,
  "model": "gemini-2.0-flash-exp"
}
```

### 2. Chat (Main API)
```bash
POST http://localhost:8002/chat

Request:
{
  "query": "Làm thế nào để cài đặt thông tin cửa hàng?",
  "conversation_id": "optional-id",
  "include_history": false
}

Response:
{
  "answer": "Để cập nhật thông tin cửa hàng trong Tendoo App, bạn làm theo các bước sau:\n\n1. Vào menu Cài đặt > Cửa hàng > Thông tin cửa hàng\n2. Điền các thông tin sau:\n   - Tên cửa hàng\n   - Địa chỉ\n   - Số điện thoại\n   - Email liên hệ\n3. Nhấn nút Lưu để hoàn tất\n\n⚠️ Lưu ý:\n- Tên cửa hàng sẽ hiển thị trên hóa đơn\n- Email sẽ được sử dụng để nhận thông báo\n\nBạn còn thắc mắc gì khác không?",
  "retrieved_chunks": [
    {
      "chunk_id": "abc123",
      "section_code": "1.1.1",
      "section_title": "Thông tin cửa hàng",
      "content": "Để cập nhật thông tin cửa hàng...",
      "score": 0.8523,
      "title_path": "Cài đặt cửa hàng > Cửa hàng > Thông tin cửa hàng"
    }
  ],
  "metadata": {
    "total_retrieved": 10,
    "total_enriched": 25,
    "total_used": 15,
    "hybrid_search": true,
    "graph_enrichment": true,
    "reranking": true
  }
}
```

### 3. Stats
```bash
GET http://localhost:8002/stats

Response:
{
  "total_chunks": 105,
  "by_section_type": {
    "section_1": 2,
    "section_2": 4,
    "section_3": 7,
    "item_number": 45,
    "item_dash": 32
  },
  "by_level": {...},
  "top_tags": {
    "cài đặt": 10,
    "bán hàng": 8
  },
  "config": {
    "llm_model": "gemini-2.0-flash-exp",
    "embedding_model": "bge-m3",
    "hybrid_search": true,
    "graph_enrichment": true,
    "reranking": true
  }
}
```

## Ví dụ sử dụng

### Python
```python
import requests

url = "http://localhost:8002/chat"
payload = {
    "query": "Tendoo hỗ trợ những phương thức thanh toán nào?"
}

response = requests.post(url, json=payload)
data = response.json()

print("Trả lời:", data["answer"])
```

### cURL
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quy trình bán hàng cho shop FnB như thế nào?"
  }'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8002/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Có những mẫu hóa đơn nào?'
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
```

## Câu hỏi mẫu

1. **Về cài đặt:**
   - "Làm thế nào để cài đặt thông tin cửa hàng?"
   - "Cách cấu hình phương thức thanh toán?"
   - "Làm sao để tạo website bán hàng?"

2. **Về bán hàng:**
   - "Quy trình bán hàng cho shop FnB như thế nào?"
   - "Các bước bán hàng cho shop bán lẻ?"
   - "Cách áp dụng khuyến mãi?"

3. **Về sản phẩm:**
   - "Thông tin sản phẩm cần có những gì?"
   - "Cách quản lý tồn kho?"
   - "Làm sao để nhập hàng vào kho?"

4. **Về hóa đơn:**
   - "Có những mẫu hóa đơn nào?"
   - "Sự khác biệt giữa các mẫu hóa đơn?"
   - "Mẫu hóa đơn nào phù hợp với shop của tôi?"

## Tùy chỉnh

### Thay đổi số lượng chunks
Trong `tendoo_chatbot.py`:
```python
class Config:
    TOP_K = 5  # Số chunks retrieve
    MAX_DESCENDANTS = 5  # Số children
    MAX_SIBLINGS = 3  # Số siblings
    RERANK_TOP_K = 3  # Số chunks sau rerank
```

### Bật/tắt tính năng
```python
class Config:
    USE_HYBRID_SEARCH = True  # Hybrid search
    USE_GRAPH_ENRICHMENT = True  # Include parent/children
    USE_RERANKING = True  # LLM reranking
```

### Thay đổi model
```python
class Config:
    LLM_MODEL = "gemini-2.0-flash-exp"  # Hoặc "gemini-1.5-pro"
    EMBEDDING_MODEL = "bge-m3"  # Hoặc model khác từ Ollama
```

### Tùy chỉnh prompt
Sửa hàm `generate_answer()` trong `tendoo_chatbot.py`:
```python
prompt = f"""Bạn là trợ lý AI...
[Sửa prompt của bạn ở đây]
"""
```

## Troubleshooting

### Lỗi: "Chunks chưa được load"
**Nguyên nhân:** File `output_tendoo/chunks.json` không tồn tại.

**Giải pháp:**
```bash
python test_tendoo.py  # Tạo chunks trước
```

### Lỗi: "Cannot connect to Ollama"
**Nguyên nhân:** Ollama chưa chạy hoặc chưa có model bge-m3.

**Giải pháp:**
```bash
# Kiểm tra Ollama
ollama list

# Pull model nếu chưa có
ollama pull bge-m3
```

### Lỗi: "Gemini API key invalid"
**Nguyên nhân:** API key không hợp lệ.

**Giải pháp:**
Cập nhật API key trong `tendoo_chatbot.py`:
```python
class Config:
    GEMINI_API_KEY = "your-api-key-here"
```

### Server chậm
**Nguyên nhân:** Embedding/LLM mất thời gian.

**Giải pháp:**
- Giảm `TOP_K` xuống 3
- Tắt `USE_RERANKING = False`
- Giảm `MAX_DESCENDANTS` và `MAX_SIBLINGS`

## Performance

### Thời gian xử lý (trung bình)
- **Embedding query**: ~100ms
- **Hybrid search**: ~50ms
- **Graph enrichment**: ~20ms
- **LLM reranking**: ~500ms (nếu bật)
- **Generate answer**: ~2-3s (Gemini)

**Tổng**: ~3-4s/query (với reranking), ~2-3s (không reranking)

### Tối ưu hóa
Để tăng tốc độ:
1. Tắt reranking: `USE_RERANKING = False`
2. Giảm số chunks: `TOP_K = 3`
3. Giảm context: `MAX_DESCENDANTS = 2`, `MAX_SIBLINGS = 1`
4. Sử dụng model nhỏ hơn cho Gemini

## License

MIT

## Contact

Nếu có vấn đề, vui lòng tạo issue hoặc liên hệ support team.
