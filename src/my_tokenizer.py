# src/my_tokenizer.py

class TTSTokenizer:
    def __init__(self, characters, pad, eos, bos, unk):
        self.characters = characters
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.unk = unk

        # Create a mapping from characters to indices
        # Reserve 0 for padding
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.characters)}
        self.char2idx[self.eos] = len(self.char2idx) + 1
        self.char2idx[self.bos] = len(self.char2idx) + 1
        self.char2idx[self.unk] = len(self.char2idx) + 1

        self.vocab_size = len(self.char2idx) + 1  # +1 for padding

    def encode(self, text):
        """
        Encodes a given text into a list of token indices.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of integers representing token indices.
        """
        return [self.char2idx.get(char, self.char2idx[self.unk]) for char in text]
