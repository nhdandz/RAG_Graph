"""
Query Expansion Configuration
==============================

Separate config file for easy maintenance and domain customization.
Edit this file to add/remove synonyms without touching the main code.

Usage:
    from query_config import QUERY_SYNONYMS
"""

# Synonym mapping for query expansion
# Format: {term: [list of synonyms]}
QUERY_SYNONYMS = {
    # ====== General enrollment terms ======
    'học viện': ['trường'],
    'thi vào': ['tuyển sinh', 'dự tuyển', 'xét tuyển'],

    # ====== Health-related terms ======
    # Note: These can be removed if not needed for your use case
    'cận thị': ['cận', 'mắt cận', 'tật khúc xạ cận thị'],
    'điốp': ['dioptre', 'độ'],

    # ====== Add more synonyms here ======
    # Example:
    # 'hồ sơ': ['giấy tờ', 'tài liệu'],
    # 'điểm chuẩn': ['điểm sàn', 'điểm trúng tuyển'],
}

# You can easily customize synonyms for different domains:
# - Remove health-related terms if not applicable
# - Add domain-specific synonyms as needed
# - Different deployments can use different config files
