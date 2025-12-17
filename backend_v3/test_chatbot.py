# -*- coding: utf-8 -*-
"""
Script test Tendoo Chatbot
"""

import requests
import json


def test_chatbot(query: str, show_details: bool = True):
    """Test chatbot with a query"""
    url = "http://localhost:8002/chat"

    payload = {
        "query": query,
        "include_history": False
    }

    print(f"\n{'='*80}")
    print(f"❓ Câu hỏi: {query}")
    print(f"{'='*80}\n")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        # Print answer
        print("🤖 Trả lời:")
        print("-" * 80)
        print(data["answer"])
        print("-" * 80)

        if show_details:
            # Print retrieved chunks
            print(f"\n📚 Retrieved Chunks ({len(data['retrieved_chunks'])}):")
            for i, chunk in enumerate(data["retrieved_chunks"], 1):
                print(f"\n  {i}. [{chunk['section_code']}] {chunk['section_title']}")
                print(f"     Path: {chunk['title_path']}")
                print(f"     Score: {chunk['score']:.4f}")
                print(f"     Content: {chunk['content'][:100]}...")

            # Print metadata
            if data.get("metadata"):
                print(f"\n📊 Metadata:")
                for key, value in data["metadata"].items():
                    print(f"  - {key}: {value}")

        return data

    except requests.exceptions.ConnectionError:
        print("❌ Lỗi: Không thể kết nối đến chatbot!")
        print("   Hãy chạy: python tendoo_chatbot.py")
        return None
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return None


def main():
    """Main test function"""
    print("\n" + "="*80)
    print("TEST TENDOO CUSTOMER SUPPORT CHATBOT")
    print("="*80)

    # Check if server is running
    try:
        response = requests.get("http://localhost:8002/")
        if response.status_code == 200:
            info = response.json()
            print(f"\n✅ Server đang chạy:")
            print(f"  - Service: {info['service']}")
            print(f"  - Chunks loaded: {info['chunks_loaded']}")
            print(f"  - Model: {info['model']}")
        else:
            print("\n❌ Server không phản hồi đúng!")
            return
    except:
        print("\n❌ Server chưa chạy!")
        print("Hãy chạy: python tendoo_chatbot.py")
        return

    # Test queries
    test_queries = [
        "Làm thế nào để cài đặt thông tin cửa hàng?",
        "Tendoo hỗ trợ những phương thức thanh toán nào?",
        "Quy trình bán hàng cho shop FnB như thế nào?",
        "Có những mẫu hóa đơn nào?",
        "Cách nhập hàng vào kho?",
    ]

    print("\n" + "="*80)
    print("BẮT ĐẦU TEST CÁC CÂU HỎI MẪU")
    print("="*80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST #{i}")
        print(f"{'#'*80}")
        test_chatbot(query, show_details=(i == 1))  # Show details for first query only

        if i < len(test_queries):
            input("\n👉 Nhấn Enter để tiếp tục...")

    print("\n\n" + "="*80)
    print("HOÀN THÀNH TEST")
    print("="*80)

    # Interactive mode
    print("\n" + "="*80)
    print("CHẾ ĐỘ TƯƠNG TÁC")
    print("="*80)
    print("Nhập câu hỏi của bạn (hoặc 'quit' để thoát):")

    while True:
        print("\n" + "-"*80)
        query = input("❓ Bạn: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Tạm biệt!")
            break

        if not query:
            continue

        test_chatbot(query, show_details=False)


if __name__ == "__main__":
    main()
