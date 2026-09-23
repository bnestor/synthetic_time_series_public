import unittest

from synthetic_time_series_clean.common.tokens import assert_single_token, ensure_special_tokens


class FakeTokenizer:
    eos_token = None
    pad_token = None
    eos_token_id = None

    def __init__(self):
        self.additional_special_tokens = []
        self.ids = {"</event>": 7}

    def add_special_tokens(self, payload):
        if "pad_token" in payload:
            self.pad_token = payload["pad_token"]
            return 1
        tokens = payload["additional_special_tokens"]
        added = 0
        for token in tokens:
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, -1)

    def encode(self, token, add_special_tokens=False):
        del add_special_tokens
        return [self.ids[token]]


class TokenTests(unittest.TestCase):
    def test_ensure_special_tokens_counts_pad_and_additional_tokens(self):
        tokenizer = FakeTokenizer()
        added = ensure_special_tokens(tokenizer, ["</event>", "<event>"])
        self.assertEqual(added, 3)

    def test_assert_single_token_returns_id(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(assert_single_token(tokenizer, "</event>"), 7)

    def test_assert_single_token_rejects_multi_token_encoding(self):
        tokenizer = FakeTokenizer()
        tokenizer.encode = lambda token, add_special_tokens=False: [1, 2]
        with self.assertRaisesRegex(RuntimeError, "single token"):
            assert_single_token(tokenizer, "</event>")


if __name__ == "__main__":
    unittest.main()
