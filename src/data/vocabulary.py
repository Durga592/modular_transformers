class Vocabulary:
    """Generic Vocabulary abstraction for mapping tokens to indices."""

    def __init__(self, pad_token="<pad>", unk_token="<unk>", bos_token="<s>", eos_token="</s>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.stoi = {pad_token: 0, unk_token: 1, bos_token: 2, eos_token: 3}
        self.itos = {0: pad_token, 1: unk_token, 2: bos_token, 3: eos_token}

    def __len__(self):
        return len(self.stoi)
