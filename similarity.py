# semantic_similarity_full.py

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz.distance import Levenshtein

from sentence_transformers import SentenceTransformer, util

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score

from rouge_score import rouge_scorer

import nltk

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")

# Optional imports
try:
    import spacy
except ImportError:
    spacy = None

try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


# =========================================================
# LEXICAL SIMILARITY METRICS
# =========================================================

def jaccard_similarity(text1, text2):

    set1 = set(text1.split())
    set2 = set(text2.split())

    if not set1 and not set2:
        return 1.0

    if not set1 or not set2:
        return 0.0

    return len(set1.intersection(set2)) / len(set1.union(set2))


def sequence_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def levenshtein_similarity(text1, text2):

    if len(text1) == 0 and len(text2) == 0:
        return 1.0

    return Levenshtein.normalized_similarity(text1, text2)


# =========================================================
# TF-IDF COSINE
# =========================================================

def compute_tfidf_cosine(df, col1, col2):

    combined = df[col1].tolist() + df[col2].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(combined)

    n = len(df)

    scores = []

    for i in range(n):

        sim = cosine_similarity(
            vectors[i],
            vectors[i + n]
        )[0][0]

        scores.append(sim)

    return scores


# =========================================================
# COUNT VECTORIZER COSINE
# =========================================================

def compute_count_cosine(df, col1, col2):

    combined = df[col1].tolist() + df[col2].tolist()

    vectorizer = CountVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(combined)

    n = len(df)

    scores = []

    for i in range(n):

        sim = cosine_similarity(
            vectors[i],
            vectors[i + n]
        )[0][0]

        scores.append(sim)

    return scores


# =========================================================
# SBERT SEMANTIC SIMILARITY
# =========================================================

def compute_sbert_similarity(
        df,
        col1,
        col2,
        model_name="all-MiniLM-L6-v2"
):

    model = SentenceTransformer(model_name)

    emb1 = model.encode(
        df[col1].tolist(),
        convert_to_tensor=True
    )

    emb2 = model.encode(
        df[col2].tolist(),
        convert_to_tensor=True
    )

    scores = []

    for i in range(len(df)):

        sim = util.cos_sim(
            emb1[i],
            emb2[i]
        ).item()

        scores.append(sim)

    return scores


# =========================================================
# SPACY EMBEDDING SIMILARITY
# =========================================================

def compute_spacy_similarity(
        df,
        col1,
        col2,
        model_name="en_core_web_md"
):

    if spacy is None:
        return [np.nan] * len(df)

    try:
        nlp = spacy.load(model_name)

    except OSError:

        print(f"spaCy model '{model_name}' not installed.")
        return [np.nan] * len(df)

    scores = []

    for _, row in df.iterrows():

        doc1 = nlp(row[col1])
        doc2 = nlp(row[col2])

        scores.append(doc1.similarity(doc2))

    return scores


# =========================================================
# APPROXIMATE WMD-LIKE SIMILARITY
# =========================================================

def compute_wmd_similarity(
        df,
        col1,
        col2,
        model_name="en_core_web_md"
):

    if spacy is None:
        return [np.nan] * len(df)

    try:
        nlp = spacy.load(model_name)

    except OSError:

        print(f"spaCy model '{model_name}' not installed.")
        return [np.nan] * len(df)

    scores = []

    for _, row in df.iterrows():

        doc1 = nlp(row[col1])
        doc2 = nlp(row[col2])

        try:

            distance = abs(doc1.vector_norm - doc2.vector_norm)

            similarity = 1 / (1 + distance)

        except Exception:

            similarity = np.nan

        scores.append(similarity)

    return scores


# =========================================================
# BERTSCORE
# =========================================================

def compute_bertscore(df, col1, col2):

    if bert_score is None:
        return [np.nan] * len(df)

    P, R, F1 = bert_score(
        df[col1].tolist(),
        df[col2].tolist(),
        lang="en",
        verbose=True
    )

    return F1.tolist()


# =========================================================
# BLEU
# =========================================================

def bleu_similarity(reference, candidate):

    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    if not ref_tokens or not cand_tokens:
        return np.nan

    smoothie = SmoothingFunction().method1

    return sentence_bleu(
        [ref_tokens],
        cand_tokens,
        smoothing_function=smoothie
    )


# =========================================================
# GLEU
# =========================================================

def gleu_similarity(reference, candidate):

    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    if not ref_tokens or not cand_tokens:
        return np.nan

    return sentence_gleu(
        [ref_tokens],
        cand_tokens
    )


# =========================================================
# METEOR
# =========================================================

def meteor_similarity(reference, candidate):

    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    if not ref_tokens or not cand_tokens:
        return np.nan

    return meteor_score(
        [ref_tokens],
        cand_tokens
    )


# =========================================================
# ROUGE
# =========================================================

def rouge_similarity(reference, candidate):

    if not reference or not candidate:

        return {
            "rouge1_f1": np.nan,
            "rouge2_f1": np.nan,
            "rougeL_f1": np.nan
        }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(reference, candidate)

    return {
        "rouge1_f1": scores["rouge1"].fmeasure,
        "rouge2_f1": scores["rouge2"].fmeasure,
        "rougeL_f1": scores["rougeL"].fmeasure
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def compute_all_similarities(
        input_file,
        col1,
        col2,
        output_file="semantic_similarity_results.csv"
):

    df = pd.read_csv(input_file)

    df[col1] = df[col1].apply(clean_text)
    df[col2] = df[col2].apply(clean_text)

    print("Computing lexical metrics...")

    df["jaccard_similarity"] = df.apply(
        lambda row: jaccard_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    df["sequence_similarity"] = df.apply(
        lambda row: sequence_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    df["levenshtein_similarity"] = df.apply(
        lambda row: levenshtein_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    print("Computing vector-space metrics...")

    df["tfidf_cosine_similarity"] = compute_tfidf_cosine(
        df,
        col1,
        col2
    )

    df["count_cosine_similarity"] = compute_count_cosine(
        df,
        col1,
        col2
    )

    print("Computing SBERT similarity...")

    df["sbert_similarity"] = compute_sbert_similarity(
        df,
        col1,
        col2
    )

    print("Computing spaCy similarity...")

    df["spacy_similarity"] = compute_spacy_similarity(
        df,
        col1,
        col2
    )

    print("Computing WMD-like similarity...")

    df["wmd_like_similarity"] = compute_wmd_similarity(
        df,
        col1,
        col2
    )

    print("Computing BERTScore...")

    df["bertscore_f1"] = compute_bertscore(
        df,
        col1,
        col2
    )

    print("Computing BLEU...")

    df["bleu_similarity"] = df.apply(
        lambda row: bleu_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    print("Computing GLEU...")

    df["gleu_similarity"] = df.apply(
        lambda row: gleu_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    print("Computing METEOR...")

    df["meteor_similarity"] = df.apply(
        lambda row: meteor_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    print("Computing ROUGE...")

    rouge_scores = df.apply(
        lambda row: rouge_similarity(
            row[col1],
            row[col2]
        ),
        axis=1
    )

    df["rouge1_f1"] = rouge_scores.apply(
        lambda x: x["rouge1_f1"]
    )

    df["rouge2_f1"] = rouge_scores.apply(
        lambda x: x["rouge2_f1"]
    )

    df["rougeL_f1"] = rouge_scores.apply(
        lambda x: x["rougeL_f1"]
    )

    # =====================================================
    # AGGREGATED SCORE
    # =====================================================

    metric_cols = [
        "jaccard_similarity",
        "sequence_similarity",
        "levenshtein_similarity",
        "tfidf_cosine_similarity",
        "count_cosine_similarity",
        "sbert_similarity",
        "spacy_similarity",
        "wmd_like_similarity",
        "bertscore_f1",
        "bleu_similarity",
        "gleu_similarity",
        "meteor_similarity",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1"
    ]

    df["average_similarity"] = df[
        metric_cols
    ].mean(
        axis=1,
        skipna=True
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    df.to_csv(output_file, index=False)

    print(f"\nResults saved to: {output_file}")

    return df


# =========================================================
# EXECUTION
# =========================================================

if __name__ == "__main__":

    input_file = "input.csv"

    # CHANGE THESE COLUMN NAMES
    text_col_1 = "text_a"
    text_col_2 = "text_b"

    output_file = "semantic_similarity_results.csv"

    results = compute_all_similarities(
        input_file=input_file,
        col1=text_col_1,
        col2=text_col_2,
        output_file=output_file
    )

    print("\nFinished!")
    print(results.head())
