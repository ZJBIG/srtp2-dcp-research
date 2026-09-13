from .base_dataset import BaseDataset
from .missing_tables import load_or_create_missing_table
import torch
import random

class HateMemesDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["hatememes_train"]
        elif split == "val":
            names = ["hatememes_dev"]
        elif split == "test":
            names = ["hatememes_test"] 

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="text",
            remove_duplicate=False,
        )
        
        # missing modality control        
        self.simulate_missing = missing_info['simulate_missing']
        missing_ratio = missing_info['ratio'][split]
        missing_type = missing_info['type'][split]    
        both_ratio = missing_info['both_ratio']
        total_num = len(self.table['image'])
        self.missing_table, self.missing_table_metadata = (
            load_or_create_missing_table(
                missing_info['missing_table_root'], names[0], split, total_num,
                missing_ratio, missing_type, both_ratio, missing_info['seed'],
                allow_legacy_seed0=missing_info.get('allow_legacy_seed0', True),
            )
        )
        print(
            'MISSING_TABLE path={} legacy={} identity={}'.format(
                self.missing_table_metadata['path'],
                self.missing_table_metadata['legacy'],
                self.missing_table_metadata['identity'],
            )
        )

    def __getitem__(self, index):
        # index -> pair data index
        # image_index -> image index in table
        # question_index -> plot index in texts of the given image
        image_index, question_index = self.index_mapper[index]
        
        # For the case of training with modality-complete data
        # Simulate missing modality with random assign the missing type of samples
        simulate_missing_type = 0
        if self.split == 'train' and self.simulate_missing and self.missing_table[image_index] == 0:
            simulate_missing_type = random.choice([0,1,2])
            
        image_tensor = self.get_image(index)["image"]
        
        # missing image, dummy image is all-one image
        if self.missing_table[image_index] == 2 or simulate_missing_type == 2:
            for idx in range(len(image_tensor)):
                image_tensor[idx] = torch.ones(image_tensor[idx].size()).float()
            
        #missing text, dummy text is ''
        if self.missing_table[image_index] == 1 or simulate_missing_type == 1:
            text = ''
        else:
            text = self.get_text(index)["text"]

        
        labels = self.table["label"][image_index].as_py()
        
        return {
            "image": image_tensor,
            "text": text,
            "label": labels,
            "missing_type": self.missing_table[image_index].item()+simulate_missing_type,
        }
