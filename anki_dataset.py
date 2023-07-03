import os
import os.path as osp
import torch
import numpy as np
import math


class ANKIDataset:
    def __init__(self, rootdir: str, split: str = "train", shuffle: bool = False, class_weight_exponent: float = 1.0):

        # import pdb; pdb.set_trace()
        data_suffix, mask_suffix = f"{split}_data.npy", f"{split}_mask.npy"
        self.source_data_files = self._crawl_directory_for_suffix(rootdir, data_suffix)
        self.source_mask_files = self._crawl_directory_for_suffix(rootdir, mask_suffix)
        self.shuffle = shuffle
        self.lengths = [np.load(f).shape[0] for f in self.source_data_files]
        self.length = sum(self.lengths)
        self.class_weight_exponent = class_weight_exponent
        self._load_file(idx=0)

        self.class_counts = [
            self._get_class_counts(torch.from_numpy(np.load(f)).float()) for f in self.source_data_files
        ]

        self.class_counts = torch.tensor(self.class_counts).sum(dim=0).tolist()
        self.class_weights = self._get_class_weights(self.class_counts)

        print("\nDataset: {} | Length: {} | Lengths: {}  \nSamples per class: {} | Class weights: {}\n".format(
            split,
            self.length,
            self.lengths,
            self.get_samples_per_class(),
            [round(w, 4) for w in self.get_class_weights()]
        ))

    def get_data_params(self):
        data_params = {
            "input_size": self.input_size,
            "output_size": 4,  # 4 classes
            "time_index": 0,
        }
        return data_params

    def get_samples_per_class(self):
        return self.class_counts

    def get_class_weights(self):
        return self.class_weights

    @staticmethod
    def _get_class_counts(source_data):
        class_counts = [0, 0, 0, 0]
        for i in range(4):
            class_counts[i] = torch.sum(source_data[:, :, 3 + i])
        return class_counts

    # @staticmethod
    def _get_class_weights(self, class_counts):
        class_weights = [0, 0, 0, 0]
        class_weights[0] = 1 / class_counts[0]
        class_weights[1] = 1 / class_counts[1]
        class_weights[2] = 1 / class_counts[2]
        class_weights[3] = 1 / class_counts[3]

        class_weights = [math.pow(w, self.class_weight_exponent) for w in class_weights]

        # Normalize
        class_weights = [w / sum(class_weights) for w in class_weights]
        return class_weights

    @staticmethod
    def _crawl_directory_for_suffix(directory, suffix):
        # Crawl the directory recursively and get all files with the given suffix
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(suffix):
                    matches.append(os.path.join(root, filename))
        return matches

    def _load_file(self, idx=0):

        self.file_idx = idx

        # D=8: time spent, lastIvl, note_length, ease_onehot + 1 (for padding)
        self.source_data = torch.from_numpy(
            np.load(self.source_data_files[self.file_idx])
        ).float()  # [N, T, D]

        self.target_data = torch.from_numpy(
            np.argmax(np.load(self.source_data_files[self.file_idx])[:, :, 3:], axis=-1)
        ).long()  # [N, T] in [0, 4] (4: padding)

        self.input_size = self.source_data.shape[-1]

        self.mask = torch.from_numpy(
            np.load(self.source_mask_files[self.file_idx])
        ).float()  # [N, T]

        # shuffle the data
        if self.shuffle:
            self.shuffle_ids = torch.randperm(self.source_data.shape[0])
            self.source_data = self.source_data[self.shuffle_ids]
            self.target_data = self.target_data[self.shuffle_ids]
            self.mask = self.mask[self.shuffle_ids]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        if idx >= self.length:
            raise IndexError("Index out of bounds")

        # Find the first index that is greater than idx [3, 6] > 3 -> 1
        cumsum = np.cumsum(self.lengths)
        file_idx = np.argmax(cumsum > idx)
        if file_idx != self.file_idx:
            self._load_file(file_idx)

        # update idx to be relative to the new file
        idx = idx - cumsum[file_idx - 1] if file_idx > 0 else idx

        source = self.source_data[idx, :-1, :]  # [T-1, D]
        target = self.target_data[idx, 1:]  # [T-1]
        mask = self.mask[idx, 1:]  # [T-1]

        return {"source": source,
                "target": target,
                "mask": mask}