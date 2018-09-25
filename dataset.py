import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random

from config import LABEL_MAPPING_2_CLASS, LABEL_MAPPING_3_CLASS, LABEL_MAPPING_2_CLASS2


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

    @property
    def offset(self):
        return int(self._data[3])

    @property
    def reverse(self):
        return int(self._data[4])


class ViratDataSet(data.Dataset):
    def __init__(self, root_path, list_file, new_length=64, modality='RGB', transform=None,
                 test_mode=False, reverse=False, mapping=None):

        if modality not in ('RGB', 'Flow'):
            raise ValueError('Modality must be RGB or Flow!')

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.test_mode = test_mode
        # For flow
        self.reverse = reverse
        self.mapping = mapping

        self._parse_list()

    def _load_image(self, record, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(self.root_path, record.path, '{}.jpg'.format(idx + record.offset))).convert(
                'RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-x.jpg'.format(idx + record.offset))).convert('L')
            y_img = Image.open(
                os.path.join(self.root_path, record.path, '{}-y.jpg'.format(idx + record.offset))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        Sample load indices
        :param record: VideoRecord
        :return: list
        """
        frame_indices = list(range(record.num_frames))
        rand_end = max(0, len(frame_indices) - self.new_length - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.new_length, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def _get_val_indices(self, record):
        frame_indices = list(range(record.num_frames))
        out = frame_indices[:self.new_length]

        for index in out:
            if len(out) >= self.new_length:
                break
            out.append(index)

        return out

    def __getitem__(self, index):
        record = self.video_list[index]

        # It seems TSN didn't sue validate data set,
        # our val data set is equivalent to TSN model's test set
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
        else:
            segment_indices = self._get_val_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for idx in indices:
            images.extend(self._load_image(record, idx))

        # For flow, reverse image for data augmentation
        # reverse_flag = random.random() >= 0.5
        # and record.reverse
        # if self.reverse and reverse_flag:
        #     images = images[::-1]

        process_data = self.transform(images)
        # ToDo: just for 2 classify
        # 1 if record.label > 0 else 0 LABEL_MAPPING_2_CLASS
        if self.mapping:
            label = self.mapping[record.label]
        else:
            label = record.label

        return process_data, label

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    pass

