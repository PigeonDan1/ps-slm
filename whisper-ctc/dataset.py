import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import kaldiio

class JsonlCTCDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_extractor, max_input_length=3000):
        self.samples = []
        with open(jsonl_file) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_input_length = max_input_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        text = sample["target"]

        if path.endswith(".wav"):
            wav, sr = torchaudio.load(path)
            wav = wav.squeeze(0).numpy()
        else:
            sr, wav = kaldiio.load_mat(path)
        if sr != 16000:
            print(f"{path}'s sampling rate is not 16khz.")

        # Whisper feature extractor
        input_features = self.feature_extractor(
            wav,
            sampling_rate=16000
        ).input_features[0]

        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(text, add_special_tokens=False).input_ids

        return {
            "input_features": torch.tensor(input_features, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def ctc_collate_fn(batch):
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_features_padded = pad_sequence(input_features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    input_lengths = torch.tensor([len(x) for x in input_features])
    label_lengths = torch.tensor([len(x) for x in labels])

    return {
        "input_features": input_features_padded,
        "input_lengths": input_lengths,
        "labels": labels_padded,
        "label_lengths": label_lengths,
    }
