# -*- coding: utf-8 -*-
"""
Script chạy toàn bộ pipeline Tendoo RAG
"""

import os
import sys
import subprocess


def print_header(text):
    """In header đẹp"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def run_command(command, description):
    """Chạy command và xử lý lỗi"""
    print(f"▶ {description}")
    print(f"  Command: {command}\n")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Lỗi: {result.stderr}")
        return False
    else:
        if result.stdout:
            print(result.stdout)
        print(f"✅ Hoàn thành: {description}\n")
        return True


def check_file_exists(filepath):
    """Kiểm tra file tồn tại"""
    if os.path.exists(filepath):
        print(f"✅ File tồn tại: {filepath}")
        return True
    else:
        print(f"❌ File không tồn tại: {filepath}")
        return False


def main():
    """Main pipeline"""

    print_header("TENDOO RAG PIPELINE")

    # Bước 1: Kiểm tra các file cần thiết
    print_header("BƯỚC 1: KIỂM TRA MÔI TRƯỜNG")

    required_files = [
        "tendoo_rag_chunking.py",
        "create_tendoo_sample.py",
        "test_tendoo.py",
        "tendoo_rag.py"
    ]

    all_exist = True
    for file in required_files:
        if not check_file_exists(file):
            all_exist = False

    if not all_exist:
        print("\n❌ Thiếu các file cần thiết. Vui lòng kiểm tra lại!")
        return

    # Bước 2: Tạo file mẫu (nếu chưa có)
    print_header("BƯỚC 2: TẠO FILE TÀI LIỆU MẪU")

    if not os.path.exists("tendoo_guide.docx"):
        if not run_command("python create_tendoo_sample.py", "Tạo file tendoo_guide.docx"):
            print("\n❌ Không thể tạo file mẫu!")
            return
    else:
        print("✅ File tendoo_guide.docx đã tồn tại, bỏ qua bước này.\n")

    # Bước 3: Chạy chunking
    print_header("BƯỚC 3: CHUNKING TÀI LIỆU")

    if not run_command("python test_tendoo.py", "Parse và chunk tài liệu"):
        print("\n❌ Chunking thất bại!")
        return

    # Bước 4: Kiểm tra output
    print_header("BƯỚC 4: KIỂM TRA KẾT QUẢ CHUNKING")

    output_dir = "output_tendoo"
    if os.path.exists(output_dir):
        chunks_file = os.path.join(output_dir, "chunks.json")
        graph_file = os.path.join(output_dir, "hierarchy_graph.gexf")

        check_file_exists(chunks_file)
        check_file_exists(graph_file)

        if os.path.exists(chunks_file):
            import json
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"📊 Tổng số chunks: {len(chunks)}")
        print()
    else:
        print(f"❌ Thư mục {output_dir} không tồn tại!")
        return

    # Bước 5: Hướng dẫn khởi động API
    print_header("BƯỚC 5: KHỞI ĐỘNG RAG API")

    print("Để khởi động RAG API, chạy lệnh sau trong terminal mới:")
    print()
    print("  python tendoo_rag.py")
    print()
    print("Hoặc:")
    print()
    print("  uvicorn tendoo_rag:app --reload --port 8001")
    print()
    print("API sẽ chạy tại: http://localhost:8001")
    print()

    # Bước 6: Hướng dẫn test query
    print_header("BƯỚC 6: TEST QUERY")

    print("Sau khi API đã khởi động, bạn có thể test query bằng:")
    print()
    print("1. Mở trình duyệt và truy cập:")
    print("   http://localhost:8001/")
    print()
    print("2. Hoặc sử dụng curl:")
    print()
    print('   curl -X POST http://localhost:8001/query \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"query": "Làm thế nào để cài đặt thông tin cửa hàng?", "topK": 5}\'')
    print()
    print("3. Xem thống kê:")
    print("   curl http://localhost:8001/stats")
    print()

    # Bước 7: Kết thúc
    print_header("HOÀN THÀNH!")

    print("✅ Pipeline đã chạy thành công!")
    print()
    print("📁 Kết quả:")
    print(f"   - Tài liệu: tendoo_guide.docx")
    print(f"   - Chunks: {output_dir}/chunks.json")
    print(f"   - Graph: {output_dir}/hierarchy_graph.gexf")
    print()
    print("📖 Xem thêm hướng dẫn chi tiết tại: README_TENDOO.md")
    print()


if __name__ == "__main__":
    main()
