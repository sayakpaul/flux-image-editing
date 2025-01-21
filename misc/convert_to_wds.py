"""
Make sure to change `path` as needed.
Install `smart_open`, `ray` before running the script.
if you're serializing to an S3 bucket, make sure you're authenticated.
"""

from datasets import Dataset 
import json
import webdataset as wds 
from smart_open import smart_open
import os
import ray
import glob

ray.init(num_cpus=16)


if __name__ == "__main__":
    path = "/fsx/sayak/.cache/datasets--TIGER-Lab--OmniEdit-Filtered-1.2M/snapshots/82455c6cd66db7f0e5bfce8d7a236441af59d6df/data/"
    all_parquets = sorted(glob.glob(f"{path}/train-*.parquet"))

    @ray.remote
    def convert_to_wds(parquet_path):
        dataset = Dataset.from_parquet(parquet_path, split="train", cache_dir=path)
        shard_path = os.path.basename(parquet_path).replace(".parquet", ".tar")
        shard_path = os.path.join("s3://omniedit-wds", shard_path)

        with smart_open(shard_path, "wb") as s3_file:
            with wds.TarWriter(s3_file) as shard_writer:
                for i, example in enumerate(dataset):
                    json_data = json.dumps(example["edited_prompt_list"]).encode("utf-8")
                    src_img = example["src_img"].convert("RGB")
                    edited_img = example["edited_img"].convert("RGB")

                    wds_example = {
                        "__key__": str(i),
                        "omni_edit_id.txt": example["omni_edit_id"],
                        "task.txt": example["task"],
                        "src_img.jpg": src_img,
                        "edited_img.jpg": edited_img,
                        "edited_prompt_list.json": json_data,
                        "sc_reasoning.txt": example["sc_reasoning"],
                        "pq_reasoning.txt": example["pq_reasoning"],
                        "height.txt": str(example["height"]),
                        "width.txt": str(example["width"]),
                        "sc_score_1.txt": str(example["sc_score_1"]),
                        "sc_score_2.txt": str(example["sc_score_2"]),
                        "pq_score.txt": str(example["pq_score"]),
                        "o_score.txt": str(example["o_score"]),
                    }
                    shard_writer.write(wds_example)

        return shard_path

    futures = [convert_to_wds.remote(path) for path in all_parquets]
    ray.get(futures)
