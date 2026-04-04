class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }
        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>"
        }

    @property
    def vocab_size(self):
        return len(self.word_to_id)

    def build_vocab(self, texts):
        next_id = len(self.word_to_id)
        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = next_id
                    self.id_to_word[next_id] = word
                    next_id += 1

    def encode(self, text, add_special_tokens=False):
        ids = []

        if add_special_tokens:
            ids.append(self.word_to_id["<BOS>"])

        for word in text.split():
            ids.append(self.word_to_id.get(word, self.word_to_id["<UNK>"]))

        if add_special_tokens:
            ids.append(self.word_to_id["<EOS>"])

        return ids

    def decode(self, ids, skip_special_tokens=False):
        words = []
        special_tokens = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}

        for idx in ids:
            word = self.id_to_word.get(idx, "<UNK>")
            if skip_special_tokens and word in special_tokens:
                continue
            words.append(word)

        return " ".join(words)