import webdataset as wds
import torch 
import numpy as np
from torchvision import transforms
import math

# Taken from https://github.com/tmbdev-archive/webdataset-imagenet-2/blob/01a4ab54307b9156c527d45b6b171f88623d2dec/imagenet.py#L65.
def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
    else:
        yield from src

def collate_fn(samples):
    source_pixel_values = torch.stack([example["source_pixel_values"] for example in samples])
    source_pixel_values = source_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in samples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    captions = [example["prompt"] for example in samples]
    return {"source_pixel_values": source_pixel_values, "edited_pixel_values": edited_pixel_values, "captions": captions}


class QualityFilter:
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, x):
        avg_sc_score = (x["sc_score_1"] + x["sc_score_2"]) / 2
        select = avg_sc_score >= self.threshold and x["pq_score"] >= self.threshold and x["o_score"] >= self.threshold
        if select:
            return True
        else:
            return False

class ControlFluxDataset:
    def __init__(self, args):
        self.args = args
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    
    def get_dataset(self):
        args = self.args
        dataset = (
            wds.WebDataset(
                args.dataset_path, 
                handler=wds.warn_and_continue, 
                nodesplitter=nodesplitter, 
                shardshuffle=500,
                empty_check=False
            )
            .shuffle(2000, handler=wds.warn_and_continue)
            .decode("pil", handler=wds.warn_and_continue)
            .rename(
                src_img="src_img.jpg",
                edited_img="edited_img.jpg",
                prompt_list="edited_prompt_list.json",
                sc_score_1="sc_score_1.txt",
                sc_score_2="sc_score_2.txt",
                pq_score="pq_score.txt",
                o_score="o_score.txt",
                handler=wds.warn_and_continue,
            )
        )
        dataset = dataset.map(self.preprocess_fn, handler=wds.warn_and_continue)
        dataset = dataset.select(QualityFilter(args.quality_threshold)) if args.quality_threshold else dataset
        return dataset

    def preprocess_fn(self, sample):
        source_pixel_values = self.image_transforms(sample["src_img"])
        edited_pixel_values = self.image_transforms(sample["edited_img"])
        prompt = np.random.choice(sample["prompt_list"]) if isinstance(sample["prompt_list"], list) else sample["prompt_list"]

        return {
            "source_pixel_values": source_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompt": prompt,
            "sc_score_1": float(sample["sc_score_1"]),
            "sc_score_2": float(sample["sc_score_2"]),
            "pq_score": float(sample["pq_score"]),
            "o_score": float(sample["o_score"])
        }

    def prepare_dataloader(self, dataset):
        args = self.args
        # per dataloader worker
        num_worker_batches = math.ceil(args.num_train_examples / (args.global_batch_size * args.dataloader_num_workers))  
        dataset = dataset.batched(
            args.per_gpu_batch_size, partial=False, collation_fn=collate_fn
        ).with_epoch(num_worker_batches)
        dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return dataloader
        

if __name__ == "__main__":
    from argparse import Namespace
    
    args = Namespace(
        dataset_path="pipe:aws s3 cp s3://omniedit-wds/train-{00000..00570}-of-00571.tar -",
        num_train_examples=1203497,
        per_gpu_batch_size=8,
        global_batch_size=64,
        num_workers=4,
        resolution=256,
    )
    dataset_obj = ControlFluxDataset(args)
    dataset = dataset_obj.get_dataset()
    
    sample_count = 0
    for sample in dataset:
        print(sample.keys())
        print(sample["prompt"])
        sample_count += 1
        
    with open("dataset_actual_count.txt", "w") as f:
        f.write(str(sample_count))

    dataloader = dataset_obj.prepare_dataloader(dataset)
    for batch in dataloader:
        print(batch.keys())
        print(batch["pixel_values"].shape)
        break
