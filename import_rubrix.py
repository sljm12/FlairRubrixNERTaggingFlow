import rubrix as rb
import os
import json
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    json_dir = "./json"
    dataset_name = "military_ner"

    for dirpath, dirnames, filenames in os.walk(json_dir):
        for f in tqdm(filenames):
            records = []
            j_str = Path(dirpath, f).read_text("utf8")
            j_obj = json.loads(j_str)
            for i in j_obj:
                if len(i["predictions"]) != 0:
                    r = rb.TokenClassificationRecord(text=i["text"],
                                                     tokens=i["tokens"],
                                                     prediction=i["predictions"])
                    records.append(r)
            if len(records) != 0:
                rb.log(records, dataset_name)