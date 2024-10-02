# tests/test_tokenizer.py

import unittest
from src.my_tokenizer import TTSTokenizer

class TestTTSTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TTSTokenizer(
            characters="abcdefghijklmnopqrstuvwxyz ",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            unk="<UNK>"
        )

    def test_encode_known_characters(self):
        text = "hello world"
        tokens = self.tokenizer.encode(text)
        expected_tokens = [self.tokenizer.char_to_id[char] for char in text]
        self.assertEqual(tokens, expected_tokens, "Known characters should be correctly encoded.")

    def test_encode_unknown_characters(self):
        text = "hello, world!"
        tokens = self.tokenizer.encode(text)
        # ',' and '!' should map to <UNK>
        expected_tokens = [
            self.tokenizer.char_to_id['h'],
            self.tokenizer.char_to_id['e'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['o'],
            self.tokenizer.char_to_id['<UNK>'],  # ','
            self.tokenizer.char_to_id[' '],
            self.tokenizer.char_to_id['w'],
            self.tokenizer.char_to_id['o'],
            self.tokenizer.char_to_id['r'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['d'],
            self.tokenizer.char_to_id['<UNK>']   # '!'
        ]
        self.assertEqual(tokens, expected_tokens, "Unknown characters should be mapped to <UNK> token.")

    def test_decode_tokens(self):
        tokens = [
            self.tokenizer.char_to_id['h'],
            self.tokenizer.char_to_id['e'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['o'],
            self.tokenizer.char_to_id['<UNK>'],  # ','
            self.tokenizer.char_to_id[' '],
            self.tokenizer.char_to_id['w'],
            self.tokenizer.char_to_id['o'],
            self.tokenizer.char_to_id['r'],
            self.tokenizer.char_to_id['l'],
            self.tokenizer.char_to_id['d'],
            self.tokenizer.char_to_id['<UNK>']   # '!'
        ]
        decoded_text = self.tokenizer.decode(tokens)
        expected_text = "hello<UNK> world<UNK>"
        self.assertEqual(decoded_text, expected_text, "Tokens should be correctly decoded, including <UNK> tokens.")

if __name__ == '__main__':
    unittest.main()
