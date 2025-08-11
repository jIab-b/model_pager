from vllm.transformers_utils.tokenizer import get_tokenizer

def load_tokenizers(clip_path, t5_path):
    clip_tok = get_tokenizer(clip_path)
    t5_tok = get_tokenizer(t5_path)
    return clip_tok, t5_tok
