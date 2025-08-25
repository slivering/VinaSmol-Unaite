import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from ..hfmodel import HFModel

# TODO: consider using GlotLID FastText classifier
classification_model_name = 'qanastek/51-languages-classifier'
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

def token_language_classification(
        llm: HFModel,
        *,
        limit: int = None,
        batch_size: int = 64,
    ) -> pd.DataFrame:
    classifier_top5 = pipeline(
        "text-classification",
        model=classification_model,
        tokenizer=classification_tokenizer,
        batch_size=batch_size,
        top_k=5,
    )

    tokenizer = llm.load_tokenizer()
    vocab = tokenizer.get_vocab()
    inverse_vocab = {token: token_id for token_id, token in vocab.items()}

    token_ids = [[token_id] for token_id in vocab.values() if token_id not in tokenizer.all_special_ids]
    token_ids = token_ids[:limit]
    cleaned_tokens = tokenizer.batch_decode(token_ids)
    outputs = classifier_top5(cleaned_tokens)
    results = []

    for cleaned, [token_id], output in zip(cleaned_tokens, token_ids, outputs):
        token = inverse_vocab[token_id]
        score = output[0]['score']
        lang = output[0]['label']
        results.append([token, cleaned, token_id, lang, score])
    
    return pd.DataFrame(results, columns=['token', 'cleaned', 'token_id', 'lang', 'score'])