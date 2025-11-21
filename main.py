from utils.misc import create_metadata_jsonl

if __name__ == "__main__":
    create_metadata_jsonl(
        "data/images",
        "metadata.jsonl",
    )