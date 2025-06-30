# scripts/chunker.py

import sys
import os
import re
import json
from typing import List, Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from utils.text_cleaning import basic_clean_text

def merge_pages_from_json(json_path: str) -> str:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 페이지 번호(int) 기준으로 정렬하여 병합
    merged_text = "\n\n".join([data[k] for k in sorted(data.keys(), key=int)])
    return merged_text
