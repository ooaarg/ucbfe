# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd


class ParquetDataset(Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data(data_path)
        print("Init completed")
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        data_arrays = []
        # print(all_cols)
        for col in all_cols:
            if df[col].dtype == "object":
                array = np.array(df[col].to_list(), dtype=int)
            else:
                array = df[col].to_numpy()
            data_arrays.append(array)
            # print(col, df[col].dtype, array[:5], array.shape)
        a = np.column_stack(data_arrays)
        # print("Column are stacked.")
        return a


class ParquetDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False,
                 num_workers=1, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        # print("Ready to create self.dataset")
        self.dataset = ParquetDataset(feature_map, data_path)
        # print("Created self.dataset")
        super().__init__(dataset=self.dataset, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers,
                         collate_fn=BatchCollator(feature_map))
        print()
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map):
        self.feature_map = feature_map

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        batch_dict = dict()
        for col in all_cols:
            batch_dict[col] = batch_tensor[:, self.feature_map.get_column_index(col)]
        return batch_dict
