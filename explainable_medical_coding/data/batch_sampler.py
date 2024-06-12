"""
PyTorch has pack_padded_sequence this doesn’t work with dense layers. For sequence data with high variance in its length
the best way to minimize padding and masking within a batch is by feeding in data that is already grouped by sequence length
(while still shuffling it somewhat). Here is my current solution in numpy.
I will need to convert every function over to torch to allow it to run on the GPU and am sure there are many other
ways to optimize it further. Hope this helps others and that maybe it can become a new PyTorch Batch Sampler someday.
General approach to how it works:
Decide what your bucket boundaries for the data are.
1. Iterate through your data (provided in an array) and for each element its index and length is recorded
2. Given these indices and lengths, each index is assigned to a bucket ID (I took this whole function from the tensorflow batch_by_sequence_length linked to above)
3. Shuffle the data in these buckets
4. Split the data in each bucket into approximately the batch size (may be slightly larger)
5. Shuffle all of the batches made
6. yield a batch (which contains index references to your data)
Some code and inspiration taken from: https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length
"""


import random

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Sampler


class BySequenceLengthSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        bucket_boundaries: list[int],
        batch_size: int = 64,
        drop_last: bool = True,
    ):
        self.data_len = len(dataset)
        self.ind_n_len = [(idx, length) for idx, length in enumerate(dataset["length"])]
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print(
                "WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... "
            )

        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)

    def shuffle_tensor(self, t):
        return t[torch.randperm(len(t))]

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)

            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        random.shuffle(
            iter_list
        )  # shuffle all the batches so they arent ordered by bucket
        # size
        for batch in iter_list:
            yield batch.numpy().tolist()  # as it was stored in an array

    def __len__(self):
        return (self.data_len // self.batch_size) + 1 - int(self.drop_last)

    def element_to_bucket_id(self, x, seq_length):
        valid_buckets = (seq_length >= self.buckets_min) * (
            seq_length < self.buckets_max
        )
        bucket_id = valid_buckets.nonzero()[0].item()

        return bucket_id
