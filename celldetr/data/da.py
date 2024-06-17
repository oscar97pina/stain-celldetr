import torch

class DomainAdaptationDataset():
    def __init__(self, source, target, randperm=None):
        self.src_dataset = source
        self.tgt_dataset = target

        num_srcs = len(self.src_dataset)
        self.src_indices = torch.randperm(num_srcs, generator=torch.Generator().manual_seed(randperm))\
            if randperm else torch.arange(num_srcs)
        num_tgts = len(self.tgt_dataset)
        self.tgt_indices = torch.randperm(num_tgts, generator=torch.Generator().manual_seed(randperm))\
            if randperm else torch.arange(num_tgts)
    
    def __len__(self):
        return max(len(self.src_dataset), len(self.tgt_dataset))
    
    def __getitem__(self, idx):
        # get src images and labels
        src_idx = self.src_indices[idx % len(self.src_dataset)]
        src_imgs, src_lbls = self.src_dataset[src_idx]
        # get tgt images
        tgt_idx = self.tgt_indices[idx % len(self.tgt_dataset)]
        tgt_imgs, _, _ = self.tgt_dataset[tgt_idx]
        if isinstance(tgt_imgs, tuple):
            tgt_imgs = tgt_imgs[0]
        
        return src_imgs, src_lbls, tgt_imgs