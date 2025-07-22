import sentencepiece as spm

sp_model_path = "lang_bpe_500.model"
output_txt_path = "lang_bpe_500.txt"

sp = spm.SentencePieceProcessor()
sp.load(sp_model_path)

vocab_size = sp.get_piece_size()

with open(output_txt_path, "w", encoding="utf-8") as f:
    for idx in range(vocab_size):
        token = sp.id_to_piece(idx)
        f.write(f"{token} {idx}\n")

print(f"✅ Vocabulary saved to {output_txt_path}")
