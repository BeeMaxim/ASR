from torch import nn
from torch.nn import Sequential


class DeepSpeech2(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(1, 32, (11, 41), stride=(2, 2), padding=(5, 20), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, (11, 21), stride=(1, 2), padding=(5, 10), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 96, (11, 21), stride=(1, 2), padding=(5, 10), bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(input_size=16 * 96, hidden_size=512, num_layers=2, batch_first=True)

        self.head = Sequential(
            nn.Linear(in_features=512, out_features=n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        # print(spectrogram.transpose(1, 2).shape)
        # print('SQQUEEEEEEEEEEEEEEEEEEEEEEEZE')
        # print(spectrogram.transpose(1, 2).unsqueeze(1).shape)
        output = self.net(spectrogram.transpose(1, 2).unsqueeze(1)).transpose(1, 2).reshape(spectrogram.shape[0], -1, 16 * 96)
        # print(output.shape)

        output = self.rnn(output)[0]
        # print(output.shape)

        output = self.head(output)
        # print(output.shape)

        # print(output[:, :10, :])

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return (input_lengths + 1) // 2

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
