# 🚀 Quick Start - RAG Demo with Tendoo Chunks

Hướng dẫn nhanh để chạy RAG Demo với chunks Tendoo đã được tối ưu.

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- Ollama với models: `bge-m3`, `qwen3:14b`

## 🎯 Cách Sử Dụng Nhanh

### Option 1: Sử dụng UI (Khuyến Nghị) ⭐

#### Bước 1: Khởi động Backend

```bash
cd backend
python3 main_enhanced.py
```

Server chạy tại: `http://localhost:8080`

#### Bước 2: Khởi động Frontend

```bash
cd ragdemo
npm run dev
```

Frontend chạy tại: `http://localhost:3000`

#### Bước 3: Load Chunks vào System

1. Mở browser: `http://localhost:3000`
2. Nhấn button **"📂 Load Tendoo Chunks"** (màu tím)
3. Đợi ~30-60 giây để load 70 chunks
4. Thấy thông báo: ✅ Success! Loaded 70 chunks...

#### Bước 4: Test Query

Thử các câu hỏi sau:
- "Hướng dẫn cài đặt phương thức thanh toán"
- "Quy trình bán hàng cho shop FnB như thế nào"
- "Cách tạo sản phẩm mới"

### Option 2: Sử dụng Script Python

```bash
cd backend
python3 load_chunks_api.py
```

Script này sẽ:
- ✅ Load 70 chunks tự động
- ✅ Test với queries mẫu
- ✅ Hiển thị kết quả chi tiết

## 🎨 UI Features

### 1. Upload Document Section

**Option A: Upload File**
- Chọn file (.txt, .pdf, .docx)
- Click "Upload"
- System sẽ tự động chunk

**Option B: Load Pre-chunked Tendoo** ⭐
- Click "📂 Load Tendoo Chunks"
- Load 70 chunks đã optimize sẵn
- Nhanh hơn và chất lượng tốt hơn

### 2. Query Section

- Nhập câu hỏi tiếng Việt
- Nhấn Enter hoặc click "Ask"
- Xem kết quả với context hierarchy

### 3. Context Visualization

Hiển thị 3 loại chunks:
- **Primary Match** (xanh dương): Kết quả trực tiếp
- **Parent Context** (tím): Context cấp cao hơn
- **Related Content** (xanh lá): Nội dung liên quan

## 📊 Features

### Backend (FastAPI)
- ✅ Hybrid Search (Dense + BM25)
- ✅ Context Enrichment (Parent + Related chunks)
- ✅ Load từ JSON (pre-chunked)
- ✅ Upload document trực tiếp
- ✅ Chunks Viewer UI

### Frontend (Next.js)
- ✅ Modern UI với dark mode
- ✅ Load Tendoo chunks 1-click
- ✅ Upload files
- ✅ Expandable context cards
- ✅ Type badges (Primary/Parent/Related)

## 🔧 API Endpoints

### Load Pre-chunked JSON
```bash
POST http://localhost:8080/api/documents/load-from-json
```

### Upload Document
```bash
POST http://localhost:8080/api/documents/upload
FormData: file
```

### Query
```bash
POST http://localhost:8080/api/query
Body: {"query": "...", "topK": 3}
```

### View Chunks
```bash
GET http://localhost:8080/chunks-viewer
```

## 📁 File Structure

```
Demo/
├── backend/
│   ├── main_enhanced.py              # FastAPI server
│   ├── tendoo_chunks_final.json      # 70 chunks (213KB)
│   ├── tendoo_chunk_processor.py     # Chunk creator
│   ├── load_chunks_api.py            # Test script
│   └── test_chunks.py                # Chunk tester
│
└── ragdemo/
    └── app/
        └── page.tsx                  # Main UI (updated)
```

## ✨ New Features

### 1. One-Click Load
Click button "📂 Load Tendoo Chunks" để load chunks ngay lập tức.

### 2. Optimized Chunks
70 chunks đã được:
- ✅ Chia theo hierarchy rõ ràng
- ✅ Có metadata đầy đủ
- ✅ Độ dài tối ưu (113-5514 chars)
- ✅ Giữ nguyên context

### 3. Better UI/UX
- Divider "OR" giữa upload và load
- Status messages rõ ràng
- Loading states
- Chunk statistics

## 🎯 Use Cases

### 1. Hỗ Trợ Khách Hàng
**Query:** "Làm sao để tích hợp với ngân hàng?"

**System sẽ:**
1. Tìm chunks về "Phương thức thanh toán"
2. Thêm parent context về "Cài đặt cửa hàng"
3. Thêm related chunks về các ngân hàng cụ thể
4. Generate câu trả lời đầy đủ

### 2. Hướng Dẫn Sử Dụng
**Query:** "Quy trình bán hàng FnB"

**System sẽ:**
1. Tìm chunks về quy trình FnB
2. So sánh với quy trình bán lẻ
3. Liệt kê các bước chi tiết

### 3. Tìm Hiểu Tính Năng
**Query:** "Những tính năng nào của Tendoo?"

**System sẽ:**
1. Tổng hợp từ nhiều chunks
2. Phân loại theo module
3. Giải thích chi tiết

## 🐛 Troubleshooting

### Lỗi: Cannot connect to server
**Giải pháp:**
```bash
# Kiểm tra backend đã chạy chưa
curl http://localhost:8080/api/health
```

### Lỗi: JSON file not found
**Giải pháp:**
```bash
# Kiểm tra file có tồn tại
ls -lh backend/tendoo_chunks_final.json

# Hoặc chạy lại chunk processor
cd backend
python3 tendoo_chunk_processor.py
```

### Chunks load nhưng query không ra kết quả
**Giải pháp:**
```bash
# Pull embedding model
ollama pull bge-m3

# Pull LLM model
ollama pull qwen3:14b
```

### Frontend không hiển thị button
**Giải pháp:**
```bash
# Clear cache và rebuild
cd ragdemo
rm -rf .next
npm run dev
```

## 💡 Tips

### 1. Clear Documents Trước Khi Load Mới
```bash
curl -X DELETE http://localhost:8080/api/documents
```

### 2. Xem Chunks Đã Load
Browser: `http://localhost:8080/chunks-viewer`

### 3. Test Nhanh Với Script
```bash
cd backend
python3 test_chunks.py
```

### 4. Monitor Backend Logs
Xem console của `main_enhanced.py` để debug

## 📈 Performance

- **Load time**: ~30-60s cho 70 chunks
- **Query time**: ~2-5s (tùy model)
- **Memory**: ~500MB RAM
- **Storage**: 213KB JSON + embeddings

## 🎓 Learning Resources

1. **CHUNKING_README.md** - Chi tiết về chunking strategy
2. **INTEGRATION_README.md** - Tích hợp API
3. **HIERARCHICAL_CHUNKING_README.md** - Hierarchical chunking

## 🚀 Next Steps

1. ✅ Load chunks và test
2. ⏭️ Thử các queries khác nhau
3. ⏭️ Tùy chỉnh chunking parameters
4. ⏭️ Add more documents
5. ⏭️ Implement persistent storage

## 📞 Support

Nếu gặp vấn đề:
1. Check backend logs
2. Check browser console
3. Xem file INTEGRATION_README.md
4. Test từng component riêng lẻ

---

**Enjoy your RAG Demo!** 🎉
