import torch

from models.superpoint import SuperPoint
from models.NCTR import NCTR


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + NCTR) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.NCTR = NCTR(config.get('NCTR', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and NCTR
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        data['keypoints0'], data['keypoints1'] = data['keypoints0'].transpose(0, 1), data['keypoints1'].transpose(0, 1)
        data['descriptors0'], data['descriptors1'] = data['descriptors0'].permute(2, 0, 1), data[
            'descriptors1'].permute(2, 0, 1)
        data['scores0'], data['scores1'] = data['scores0'].transpose(0, 1), data['scores1'].transpose(0, 1)
        # Perform the matching
        pred = {**pred, **self.NCTR(data)}

        return pred
