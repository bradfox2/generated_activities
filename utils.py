from torchtext.data import Field
from torch import Tensor

def field_printer(field:Field, prob_tensor:Tensor, tgt: Tensor) -> Tensor:
    print(
        list(
            zip(
                [
                    field.vocab.itos[i]
                    for i in prob_tensor.argmax(dim=-1).flatten()
                    if field.vocab.itos[i] != "<pad>"
                ],
                [
                    field.vocab.itos[i]
                    for i in tgt.flatten()
                    if field.vocab.itos[i] != "<pad>"
                ],
            )
        )
    )