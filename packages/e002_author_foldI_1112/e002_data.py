"""Author SitsSCD data/augmentations with a verified decoded-pixel SSD backend.

The author model, loss, transforms and normalization remain unchanged. The
adapter avoids JPEG redecoding and scopes labels to explicit author AOI folds.
"""
from pathlib import Path
import copy
import os
import json
import random
import sys
import numpy as np
import torch

WORK=Path(__file__).resolve().parent
VENDOR=Path(os.environ['E002_AUTHOR_REPO']).resolve()
sys.path.insert(0,str(VENDOR))
from data.data import DynamicEarthNet, collate_fn
from data.transforms import random_fliph,random_flipv,random_rotate,random_resize_crop

RAW=Path(os.environ['E002_RAW_ROOT']).resolve()
SSD=Path(os.environ['E002_SSD_ROOT']).resolve()


class LazyCrop:
    def __init__(self,array,rows):self.array,self.rows=array,rows


class CachedAuthorDEN(DynamicEarthNet):
    def __init__(self,regions,split):
        self.explicit_regions=sorted(regions)
        original=json.loads((RAW/'split.json').read_text(encoding='utf-8'))
        assert set(self.explicit_regions)<=set(original['train'])
        assert not set(self.explicit_regions)&set(original['val']+original['test'])
        self._labels_loaded=None;self._mm={}
        # Original constructor supplies exact mean/std half tensors, monthly
        # positions and sampling defaults. Its two GT calls share one allocation.
        super().__init__(path=str(RAW),split=split,domain_shift=split!='train',
                         num_channels=4,num_classes=6,img_size=128,true_size=1024,
                         train_length=12,date_aug_range=2)
        # Author spatial fold validation AOIs live in the original source pool.
        self.image_folder=str(RAW/'train')
        index=json.loads((SSD/'index.json').read_text(encoding='utf-8'))
        self.day_pos={day:i for i,day in enumerate(index['days'])}
        assert set(self.explicit_regions)<=set(index['tiles'])

    def load_ground_truth(self,split):
        if self._labels_loaded is None:
            gt=torch.empty((len(self.explicit_regions),24,1024,1024),dtype=torch.int8)
            for i,region in enumerate(self.explicit_regions):
                p=RAW/'labels'/f'{region}.npy'
                a=np.load(p,mmap_mode='r')
                assert a.shape==(24,1024,1024) and a.dtype==np.uint8
                assert a.min()>=0 and a.max()<=6
                gt[i].copy_(torch.from_numpy(np.array(a,copy=True)).to(torch.int8))
            self._labels_loaded=gt,self.explicit_regions
        return self._labels_loaded

    def load_data(self,sits_number,sits_id,months,curr_sits_path):
        days=[self.random_date_augmentation(m) for m in months]
        if sits_id not in self._mm:
            a=np.load(SSD/f'{sits_id}.blk.npy',mmap_mode='r')
            assert a.shape==(8,8,len(self.day_pos),128,128,4) and a.dtype==np.uint8
            self._mm[sits_id]=a
        return LazyCrop(self._mm[sits_id],[self.day_pos[d] for d in days]),days

    @staticmethod
    def crop(array,rows,y,x,size):
        bi=range(y//128,(y+size-1)//128+1)
        bj=range(x//128,(x+size-1)//128+1)
        block=np.empty((len(rows),len(bi)*128,len(bj)*128,4),dtype=np.uint8)
        for i,r in enumerate(bi):
            for j,c in enumerate(bj):block[:,i*128:(i+1)*128,j*128:(j+1)*128]=array[r,c,rows]
        yy,xx=y-bi[0]*128,x-bj[0]*128
        return np.ascontiguousarray(block[:,yy:yy+size,xx:xx+size].transpose(0,3,1,2))

    def transform(self,data,gt=None,patch_loc_i=None,patch_loc_j=None):
        if self.split=='train':
            # Exact random_crop draws/order from the pinned author implementation.
            y=random.randint(0,self.true_size-self.img_size-1)
            x=random.randint(0,self.true_size-(1+2*int(y>=self.true_size-3*self.img_size))*self.img_size-1)
        else:y,x=patch_loc_i*self.img_size,patch_loc_j*self.img_size
        # Half before resize/normalization is essential: the older ST3D adapter
        # used float32 here, so it is not reused for the author-protocol run.
        z=torch.from_numpy(self.crop(data.array,data.rows,y,x,self.img_size)).to(torch.float16)
        target=gt[...,y:y+self.img_size,x:x+self.img_size]
        if self.split=='train':
            z,target=random_fliph(z,target)
            z,target=random_flipv(z,target)
            z,target=random_rotate(z,target)
            z,target=random_resize_crop(z,target)
        return z,target

    def sampling_state(self):return copy.deepcopy(self.month_list)

    def restore_sampling_state(self,state):
        assert len(state)==len(self.month_list)
        self.month_list=copy.deepcopy(state)


def verify_against_native(dataset):
    """Actual raw JPEG/author transform comparison, not an adapter self-test."""
    labels,names=dataset.gt,dataset.sits_ids

    class NativeProbe(DynamicEarthNet):
        def load_ground_truth(self,split):return labels,names

    native=NativeProbe(path=str(RAW),split='train',domain_shift=False,
                       num_channels=4,num_classes=6,img_size=128,true_size=1024,
                       train_length=12,date_aug_range=2)
    initial=dataset.sampling_state();evidence=[]
    indices=[0,len(names)//2*60,(len(names)-1)*60]
    for seed,index in zip([8101,8102,8103],indices):
        native.month_list=copy.deepcopy(initial);dataset.restore_sampling_state(initial)
        random.seed(seed);reference=native[index]
        random.seed(seed);candidate=dataset[index]
        assert reference.keys()==candidate.keys()
        assert reference['idx']==candidate['idx']
        row={'region':names[index//60],'seed':seed,'shape':list(reference['data'].shape),
             'input_dtype':str(reference['data'].dtype)}
        for key in ['data','gt','positions']:
            assert torch.equal(reference[key],candidate[key]),(key,row)
            row[key+'_exactly_equal']=True
        evidence.append(row)
    dataset.restore_sampling_state(initial)
    return evidence
