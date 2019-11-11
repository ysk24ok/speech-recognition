import torch


def pad_sequence(data):
    """
    Args:
        X (list[torch.Tensor]):
            list of tensor whose shape=(seq length, feature size)
            list length is equal to batch size
    Returns:
        torch.Tensor: shape=(padded seq length, batch size, feature size)
    """
    return torch.nn.utils.rnn.pad_sequence(data)


def pad_labels(labels):
    """
    Args:
        labels (list[torch.Tensor]):
            list of tensor whose shape=(number of labels,)
            list length is equal to batch size
    Returns:
        torch.Tensor: shape=(batch size, maximum label length)
    """
    return torch.nn.utils.rnn.pad_sequence(labels).T


def collate_for_ctc(l):
    """
    Arguments:
        l (list[tuple(torch.Tensor, torch.Tensor)]):
            list which contains tuple of
                - torch.Tensor whose shape=(sequence length, feature size)
                - torch.Tensor whose shape=(number of labels,)
            list length is equal to batch size
    Returns:
        tuple of
            - torch.Tensor
            - torch.Tensor
    """
    inputs = []
    labels = []
    input_lengths = []
    label_lengths = []
    for input, label in l:
        inputs.append(input)
        labels.append(label)
        input_lengths.append(input.shape[0])
        label_lengths.append(label.shape[0])
    ret = (
        pad_sequence(inputs),
        pad_labels(labels),
        torch.tensor(input_lengths, dtype=torch.int),
        torch.tensor(label_lengths, dtype=torch.int)
    )
    return ret
