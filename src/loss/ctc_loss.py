import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)
        '''
        print('LOOOOOOOOOOOOOOOOOOSSS')
        print(log_probs_t.shape)
        print(text_encoded.shape)
        print(log_probs_length)
        print(text_encoded_length)
        print(log_probs[0, :5, :])
        print(text_encoded[0, :])'''

        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}
