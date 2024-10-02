# src/characters_config.py

class CharactersConfig:
    """
    Handles character-related configurations for TTS.
    """

    def __init__(
        self,
        characters_class: str,
        pad: str,
        eos: str,
        bos: str,
        characters: str,
        punctuations: str,
        phonemes: str
    ):
        """
        Initializes the CharactersConfig.

        Args:
            characters_class (str): Cleaner class to process text.
            pad (str): Padding character.
            eos (str): End-of-sequence character.
            bos (str): Beginning-of-sequence character.
            characters (str): Valid characters.
            punctuations (str): Punctuation characters.
            phonemes (str): Phoneme characters (can be empty).
        """
        self.characters_class = characters_class
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.characters = characters
        self.punctuations = punctuations
        self.phonemes = phonemes

    def get_valid_characters(self) -> str:
        """
        Returns a string of valid characters for tokenization.

        Returns:
            str: Valid characters.
        """
        return self.characters + self.punctuations
