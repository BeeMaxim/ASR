import torch

from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result = {}
    result['spectrogram']  = pad_sequence([x['spectrogram'].permute(2, 1, 0).squeeze(2) for x in dataset_items]).permute(1, 2, 0)
    result['text_encoded'] = pad_sequence([x['text_encoded'].squeeze(0) for x in dataset_items]).permute(1, 0)
    result['spectrogram_length'] = torch.tensor([x['spectrogram'].shape[2] for x in dataset_items])
    result['text_encoded_length'] = torch.tensor([x['text_encoded'].shape[1] for x in dataset_items])
    result['text'] = [x['text'] for x in dataset_items]
    result['audio_path'] = [x['audio_path'] for x in dataset_items]
    result['audio'] = dataset_items[0]['audio']

    return result
