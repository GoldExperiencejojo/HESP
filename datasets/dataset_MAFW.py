import os.path
from numpy.random import randint
from torch.utils import data
import glob
import os
from datasets.video_transform import *
import numpy as np

no_image = []

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, out_data, not_out_data, transform, image_size):

        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        
        self.out = out_data 
        self.not_out = sorted(not_out_data)
        
        self._parse_list()
        pass

    def _parse_list(self):
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]

        video_list = []
        for item in tmp:
            if int(item[2]) in self.out:
                continue
            else:
                video_list.append(item)
        new_video_list = []
        label_text = []
        for i in video_list:
            i[2] = self.not_out.index(int(i[2]))
            new_video_list.append(VideoRecord(i))
            label_text.append(i[3])
        self.video_list = new_video_list
        self.label_text = label_text

        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(record.path, '*'))
        video_frames_path.sort()
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                try:
                    seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                    images.extend(seg_imgs)
                except:
                    print(record.path)
                # images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size))

        video_name = record.path.split('/')[-2] + record.path.split('/')[-1]
        return images, record.label, record._data[3], video_name 

    def __len__(self):
        return len(self.video_list)


def train_data_loader(known, unknown):
    image_size = 224
    train_transforms = torchvision.transforms.Compose([GroupRandomHorizontalFlip(),
                                                       GroupResize(image_size),
                                                       GroupRandomSizedCrop(image_size),
                                                       Stack(),
                                                       ToTorchFormatTensor()
                                                       ])
    train_data = VideoDataset(list_file="./annotation_text_label/MAFW_train.txt",
                              num_segments=8,
                              duration=1,
                              mode='train',
                              out_data=unknown,
                              not_out_data=known,
                              transform=train_transforms,
                              image_size=image_size)
    return train_data


def test_data_loader(known, unknown):
    image_size = 224
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor(),
                                                     ])
    test_data = VideoDataset(list_file="./annotation_text_label/MAFW_val.txt",
                             num_segments=8,
                             duration=1,
                             mode='test',
                             out_data=unknown,
                             not_out_data=known,
                             transform=test_transform,
                             image_size=image_size)
    return test_data

def out_data_loader(known, unknown):
    image_size = 224
    out_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor(),
                                                     ])
    out_data = VideoDataset(list_file="./annotation_text_label/MAFW_val.txt",
                             num_segments=8,
                             duration=1,
                             mode='test',
                             out_data=known, 
                             not_out_data=unknown,
                             transform=out_transform,
                             image_size=image_size)
    return out_data
