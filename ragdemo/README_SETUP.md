# RAG Demo - Hướng dẫn cài đặt và chạy

Demo RAG (Retrieval-Augmented Generation) với backend Java Spring Boot và frontend NextJS, sử dụng Ollama với BGE-M3 embedding model.

## Kiến trúc

- **Frontend**: NextJS 14 + TailwindCSS
- **Backend**: Java Spring Boot 3.2 + Maven
- **LLM/Embedding**: Ollama (BGE-M3 cho embedding, LLaMA 3.2 cho chat)
- **Vector Store**: In-memory với Cosine Similarity

## Yêu cầu hệ thống

- Java 17+
- Node.js 18+
- Maven 3.6+
- Ollama
- 8GB RAM minimum (khuyến nghị 16GB)

## Bước 1: Cài đặt Ollama

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
brew install ollama
```

### Windows
Tải từ https://ollama.com/download

## Bước 2: Pull Models

```bash
# Chạy Ollama server
ollama serve

# Trong terminal khác, pull models
ollama pull bge-m3
ollama pull llama3.2

# Verify models
ollama list
```

**Lưu ý**: BGE-M3 model khoảng 2.2GB, LLaMA 3.2 khoảng 2GB.

## Bước 3: Chạy Backend

```bash
cd backend

# Build project
mvn clean install

# Chạy application
mvn spring-boot:run
```

Backend sẽ chạy tại: **http://localhost:8080**

Kiểm tra health: `curl http://localhost:8080/api/health`

## Bước 4: Chạy Frontend

```bash
cd ragdemo

# Cài đặt dependencies (nếu chưa)
pnpm install
# hoặc: npm install

# Chạy dev server
pnpm dev
# hoặc: npm run dev
```

Frontend sẽ chạy tại: **http://localhost:3000**

## Bước 5: Sử dụng

1. Mở browser tại http://localhost:3000
2. Upload document (hỗ trợ .txt, .pdf, .docx)
3. Đợi document được index
4. Nhập câu hỏi và nhấn "Ask"
5. Xem kết quả và retrieved context

## Test API trực tiếp

### Upload document
```bash
curl -X POST http://localhost:8080/api/documents/upload \
  -F "file=@/path/to/your/document.txt"
```

### Query
```bash
curl -X POST http://localhost:8080/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "topK": 3
  }'
```

### Get document count
```bash
curl http://localhost:8080/api/documents/count
```

## Cấu trúc project

```
ragdemo/
├── backend/                 # Java Spring Boot backend
│   ├── src/
│   │   └── main/
│   │       ├── java/com/example/ragdemo/
│   │       │   ├── controller/      # REST Controllers
│   │       │   ├── service/         # Business Logic
│   │       │   ├── model/           # Data Models
│   │       │   └── config/          # Configuration
│   │       └── resources/
│   │           └── application.properties
│   └── pom.xml
├── app/                     # NextJS frontend
│   └── page.tsx            # Main UI
└── README_SETUP.md         # File này

```

## Troubleshooting

### Backend không start được

1. Kiểm tra Java version:
```bash
java -version  # Phải >= 17
```

2. Kiểm tra port 8080 có bị chiếm:
```bash
lsof -i :8080
# hoặc
netstat -an | grep 8080
```

### Ollama connection error

1. Đảm bảo Ollama đang chạy:
```bash
ps aux | grep ollama
```

2. Test Ollama API:
```bash
curl http://localhost:11434/api/version
```

3. Restart Ollama:
```bash
# Linux/macOS
killall ollama
ollama serve
```

### Frontend không kết nối được backend

1. Kiểm tra CORS settings trong backend
2. Verify backend đang chạy ở port 8080
3. Kiểm tra browser console để xem error messages

### Model generation chậm

- BGE-M3 và LLaMA 3.2 yêu cầu khá nhiều tài nguyên
- Lần đầu tiên sẽ chậm hơn (model loading)
- Nếu quá chậm, có thể đổi sang model nhỏ hơn:
  ```bash
  ollama pull llama3.2:1b  # Lightweight version
  ```
  Và update trong `application.properties`:
  ```properties
  ollama.chat-model=llama3.2:1b
  ```

### Out of memory

Tăng Java heap size:
```bash
export MAVEN_OPTS="-Xmx4g"
mvn spring-boot:run
```

Hoặc khi chạy JAR:
```bash
java -Xmx4g -jar target/ragdemo-1.0.0.jar
```

## Tính năng

- ✅ Upload và index documents (TXT, PDF, DOCX)
- ✅ Text chunking với overlap
- ✅ Vector embeddings với BGE-M3
- ✅ Semantic search với cosine similarity
- ✅ Context-aware answer generation
- ✅ Retrieved documents display với similarity scores
- ✅ In-memory vector store

## Cải tiến có thể làm

- [ ] Persistent vector store (ChromaDB, Pinecone, Weaviate)
- [ ] Streaming responses
- [ ] Chat history
- [ ] Multi-language support
- [ ] Advanced chunking strategies
- [ ] Re-ranking retrieved documents
- [ ] Document metadata filtering

## License

MIT
