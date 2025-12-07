import json
from datasets import load_dataset

# 1. 指定数据集信息
DATASET_NAME = 'wangrui6/Zhihu-KOL'
SPLIT_NAME = 'train'  # split名称为 'train'
OUTPUT_FILE = 'Zhihu-KOL-custom.jsonl'


def extract_qid(metadata_str):
    """尝试从 METADATA 字符串中解析 question_id 作为 qid。"""
    if not metadata_str:
        return ""
    try:
        # METADATA 字段是一个 JSON 字符串，包含 question_id
        metadata_dict = json.loads(metadata_str)
        qid = metadata_dict.get("question_id")

        if qid is not None:
            # 确保 qid 是字符串类型，移除可能的浮点数后缀 .0
            return str(int(qid)) if isinstance(qid, float) and qid.is_integer() else str(qid)

        return ""
    except json.JSONDecodeError:
        # 如果 METADATA 不是有效的 JSON 格式
        return ""
    except Exception:
        return ""


def convert_to_jsonl(dataset_name, split_name, output_file):
    """
    加载 Hugging Face 数据集，并将其转换为包含所有指定字段的 JSONL 文件。
    """
    try:
        # 加载数据集的特定 split
        print(f"Loading dataset: {dataset_name} (split: {split_name})...")
        # 由于数据集较大，使用流式加载可能更高效，但这里保持简单加载
        dataset = load_dataset(dataset_name, split=split_name)
        print(f"Dataset loaded. Total rows: {len(dataset)}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. 定义要保留和转换的字段
    print(f"Converting and writing to {output_file}...")
    i=0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in dataset:
            # 从 METADATA 提取 qid
            qid_value = extract_qid(row.get('METADATA', ''))

            # 构建新的字典，包含所有要求的新旧字段

            if len(row.get('RESPONSE', ''))<1024:
                i+=1
                new_record = {
                    # 新要求的字段
                    "qid": qid_value,
                    "category": "",  # 您要求设置为空字符串
                    "title": row.get('INSTRUCTION', ''),  # 您要求与 INSTRUCTION 相同
                    "desc": "",  # 您要求设置为空字符串
                    "answer": row.get('RESPONSE', ''),  # 您要求与 RESPONSE 相同
                    "SOURCE": row.get('SOURCE', ''),
                    "METADATA": row.get('METADATA', ''),
                    "upvotes": row.get('upvotes', None)
                }

                # 序列化为 JSON 字符串并写入文件
                f.write(json.dumps(new_record, ensure_ascii=False) + '\n')

    print(f"Conversion complete! Data saved to {output_file}")
    print(f"Total rows: {i}")


# 3. 执行转换
convert_to_jsonl(DATASET_NAME, SPLIT_NAME, OUTPUT_FILE)
