import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from transformers import PreTrainedTokenizer


class SNPTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = None):
        # Set up your attributes before calling super().__init__()
        self.__token_ids = vocab
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
        super().__init__(max_len=max_len)

    def _tokenize(self, text: str, **kwargs):
        return text.split(" ")

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ""
        vocab_path = Path(save_directory, filename_prefix + "vocab.json")
        json.dump(self.__token_ids, open(vocab_path, "w"))
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)
