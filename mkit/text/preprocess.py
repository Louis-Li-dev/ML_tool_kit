import re
import string
import os
import pickle
import pandas as pd
import numpy as np
from typing import Union, List, Callable

BASE_DIR = os.path.dirname(__file__)

class BaseTokenizer:
    """
    Base class for text tokenizers providing:
      - Vocabulary creation (token -> index mapping).
      - Inverse mapping (index -> token).
      - Splitting logic for single vs. multiple input strings.
      - A decode() method that raises NotImplementedError by default.
      - Handling of unseen tokens (via unknown_handler).
    """

    def __init__(
        self, 
        output_format: str = "series",
        unknown_handler: Callable[[str], object] = None
    ):
        """
        Parameters
        ----------
        output_format : str
            Either 'series' (return a pandas.Series) or 'array' (return a numpy.array).
        unknown_handler : Callable[[str], Any], optional
            A function that takes the unseen token as input, and returns an appropriate 
            representation (e.g., -1 for label encoding, or a 0-vector for one-hot).
            If not provided, child classes define a default approach.

        """
        valid_outputs = ["series", "array"]
        if output_format not in valid_outputs:
            raise ValueError(f"output_format must be one of {valid_outputs}")

        self.output_format = output_format
        self.mapping_ = {}         # token -> int
        self.inverse_mapping_ = {} # int   -> token

        # This will be used if the child class doesn't define its own default.
        self.unknown_handler = unknown_handler

    def fit(self, input_data: Union[str, List[str]]):
        """
        Collects all unique tokens from input_data to build self.mapping_ and self.inverse_mapping_.
        """
        if isinstance(input_data, str):
            input_data = [input_data]

        # Ensure all elements are strings
        if not all(isinstance(x, str) for x in input_data):
            raise ValueError("All elements of input_data must be strings.")

        # Collect all tokens
        all_tokens = []
        for text in input_data:
            all_tokens.extend(text.split())

        # Sort tokens to get a deterministic order (optional)
        unique_tokens = sorted(set(all_tokens))

        # Build the mapping (token -> index)
        self.mapping_ = {token: i for i, token in enumerate(unique_tokens)}
        # Build the inverse mapping (index -> token)
        self.inverse_mapping_ = {i: token for token, i in self.mapping_.items()}

    def transform(self, input_data: Union[str, List[str]]):
        """
        Transforms input_data into encoded form, using whichever encoding method 
        is implemented by the child class (label or one-hot).
        """
        single_input = False
        if isinstance(input_data, str):
            input_data = [input_data]
            single_input = True

        encoded_list = []
        for text in input_data:
            encoded_list.append(self._encode_string(text))

        # If the user originally passed in a single string, simplify the result
        if single_input and len(encoded_list) == 1:
            encoded_list = encoded_list[0]

        if self.output_format == "series":
            return pd.Series(encoded_list)
        else:
            return np.array(encoded_list, dtype=object)

    def fit_transform(self, input_data: Union[str, List[str]]):
        """
        Convenience method that calls fit() and then transform().
        """
        self.fit(input_data)
        return self.transform(input_data)

    def decode(self, encoded_data):
        """
        Decodes encoded data back to the original tokens.
        By default, this base class raises NotImplementedError.
        Child classes should override or implement their own decode methods.
        """
        raise NotImplementedError("decode() is not implemented in BaseTokenizer.")

    def _encode_string(self, text: str):
        """
        Abstract method for encoding a single string. 
        Child classes must implement this for label or one-hot.
        """
        raise NotImplementedError("_encode_string() must be implemented in a child class.")


class LabelTokenizer(BaseTokenizer):
    """
    Child class specialized in label encoding.
    By default, unseen tokens map to -1, or to the result of unknown_handler(token).
    >>>  LabelTokenizer(unknown_handler=lambda x: -1)   
    """
    def __init__(
        self,
        output_format: str = "series",
        unknown_handler: Callable[[str], int] = None
    ):
        super().__init__(output_format=output_format, unknown_handler=unknown_handler)
        # If user didn't provide a handler, use a default that returns -1
        if self.unknown_handler is None:
            self.unknown_handler = lambda token: -1

    def _encode_string(self, text: str):
        """
        Convert a string into integer labels according to self.mapping_.
        Unseen tokens get the unknown_handler(token) value (by default, -1).
        """
        tokens = text.split()
        output = []
        for token in tokens:
            if token in self.mapping_:
                output.append(self.mapping_[token])
            else:
                output.append(self.unknown_handler(token))
        return output

    def decode(self, encoded_data):
        """
        Decodes a list of integer labels (or list of lists) back to tokens.
        Any label == -1 (or any unknown label) is not found in inverse_mapping_
        => we might handle it by returning a placeholder (e.g. "<UNK>").
        """
        def decode_sequence(seq: List[int]):
            decoded_tokens = []
            for idx in seq:
                if idx in self.inverse_mapping_:
                    decoded_tokens.append(self.inverse_mapping_[idx])
                else:
                    # This handles -1 or any out-of-range
                    decoded_tokens.append("<UNK>")
            return decoded_tokens

        # Single list of ints?
        if isinstance(encoded_data, list) and all(isinstance(x, int) for x in encoded_data):
            return decode_sequence(encoded_data)

        # Multiple sequences
        if isinstance(encoded_data, list) and all(isinstance(x, list) for x in encoded_data):
            return [decode_sequence(seq) for seq in encoded_data]

        raise ValueError("encoded_data must be a list of int labels or a list of lists of int labels.")


class OneHotTokenizer(BaseTokenizer):
    """
    Child class specialized in one-hot encoding.
    By default, unseen tokens map to a zero vector, or you can provide a custom function.
    
    >>>  OneHotTokenizer(unknown_handler=lambda x: 0) # not recommended  
    """
    def __init__(
        self,
        output_format: str = "series",
        unknown_handler: Callable[[str], List[int]] = None
    ):
        super().__init__(output_format=output_format, unknown_handler=unknown_handler)
        # We'll define a default that returns [0,0,...,0] 
        # (the same length as vocab, but that length is only known after fit()).
        # So we create a placeholder here and finalize it in _encode_string.
        if self.unknown_handler is None:
            # We'll finalize the actual size only after we know the vocab_size
            self.unknown_handler = lambda token: None

    def _encode_string(self, text: str):
        """
        Convert a string into a list of one-hot vectors based on self.mapping_.
        Unseen tokens use unknown_handler(token), which by default is all zeros.
        """
        tokens = text.split()
        vocab_size = len(self.mapping_)

        output = []
        for token in tokens:
            if token in self.mapping_:
                vector = [0] * vocab_size
                idx = self.mapping_[token]
                vector[idx] = 1
                output.append(vector)
            else:
                # If the unknown_handler is the default (None), produce a zero vector
                if self.unknown_handler(token) is None:
                    output.append([0] * vocab_size)
                else:
                    # If the user gave a custom function, call it
                    output.append(self.unknown_handler(token))
        return output

    def decode(self, encoded_data):
        """
        Decodes one-hot encoded data back to tokens. 
        - If a vector sums to 0, that implies an unknown token, so we return "<UNK>".
        - If it sums to >1 or <1, it's invalid.
        """
        def decode_sequence(seq_of_vectors):
            tokens = []
            for vec in seq_of_vectors:
                s = sum(vec)
                if s == 1:
                    idx = vec.index(1)
                    tokens.append(self.inverse_mapping_[idx])
                elif s == 0:
                    tokens.append("<UNK>")
                else:
                    raise ValueError(f"Invalid one-hot vector: {vec} (sum != 1).")
            return tokens

        # Single sequence?
        if isinstance(encoded_data, list) and encoded_data and isinstance(encoded_data[0], list):
            if all(isinstance(x, int) for x in encoded_data[0]):
                # Single sequence of vectors
                return decode_sequence(encoded_data)
            else:
                # Multiple sequences
                return [decode_sequence(seq) for seq in encoded_data]

        raise ValueError("encoded_data must be a list of one-hot vectors or a list of lists of one-hot vectors.")


class Tokenizer:
    """
    A unified interface that can create either a LabelTokenizer or OneHotTokenizer
    based on _type. It also allows for a custom unknown_handler for unseen tokens.
    >>> Tokenizer(unknown_handler=lambda x: -1)
    """

    def __init__(
        self, 
        _type: str = "label", 
        output_format: str = "series", 
        unknown_handler=None
    ):
        valid_types = ["label", "one_hot"]
        if _type not in valid_types:
            raise ValueError(f"_type must be one of {valid_types}")

        if _type == "label":
            self.tokenizer = LabelTokenizer(
                output_format=output_format, 
                unknown_handler=unknown_handler
            )
        else:  # one_hot
            self.tokenizer = OneHotTokenizer(
                output_format=output_format, 
                unknown_handler=unknown_handler
            )

    def fit(self, input_data):
        return self.tokenizer.fit(input_data)

    def transform(self, input_data):
        return self.tokenizer.transform(input_data)

    def fit_transform(self, input_data):
        return self.tokenizer.fit_transform(input_data)

    def decode(self, encoded_data):
        return self.tokenizer.decode(encoded_data)



def clean_text(input_data: Union[str, List[str]], remove_stopwords: bool = True) -> Union[str, List[str]]:
    """
    Cleans text by:
        1. Removing non-alphabetic characters (except accented letters in À-ú range) 
        2. Converting to lowercase
        3. Removing punctuation
        4. Removing extra spaces
        5. Optionally removing stopwords (if 'remove_stopwords' is True)

    Parameters
    ----------
    input_data : Union[str, List[str]]
        The original text or list of texts to be cleaned.

    remove_stopwords : bool, optional
        Whether to remove stopwords. Defaults to True.

    Returns
    -------
    Union[str, List[str]]
        The cleaned text (str) if 'input_data' was a single string,
        otherwise a list of cleaned texts if 'input_data' was a list of strings.
    """

    def _clean_single_text(text: str) -> str:
        """Helper function to clean a single string."""

        # 1. Remove any characters that aren't A-Z, a-z, À-ú, or spaces
        processed_string = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)

        # 2. Convert all text to lowercase
        processed_string = processed_string.lower()

        # 3. Remove punctuation
        processed_string = processed_string.translate(str.maketrans('', '', string.punctuation))

        # 4. Collapse multiple spaces into a single space and strip leading/trailing spaces
        processed_string = re.sub(r'\s+', ' ', processed_string).strip()

        # 5. Remove stopwords (if requested)
        if remove_stopwords:
            path = os.path.join(BASE_DIR, './data_base/stop_words_en.pkl')
            with open(path, 'rb') as file:
                stopwords_list = pickle.load(file)

            word_list = processed_string.split()
            word_list = [token for token in word_list if token not in stopwords_list]
            processed_string = ' '.join(word_list)

        return processed_string

    # If the input is a single string, just clean and return
    if isinstance(input_data, str):
        return _clean_single_text(input_data)

    # If the input is a list, ensure all elements are strings and then clean each item
    elif isinstance(input_data, list):
        if not all(isinstance(item, str) for item in input_data):
            raise ValueError("All elements in the input list must be strings.")
        return [_clean_single_text(item) for item in input_data]

    else:
        raise ValueError("input_data must be either a string or a list of strings.")