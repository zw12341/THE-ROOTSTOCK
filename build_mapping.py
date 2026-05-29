"""
THE ROOTSTOCK — Codon-to-Word Mapping Builder
==============================================
Offline preprocessing script. Run once to generate codon_word_mapping.json,
which is then loaded at runtime by rootstock.py.

This script bridges molecular biology and poetic language by:
  1. Fetching real Arabidopsis thaliana gene sequences from NCBI
  2. Computing codon usage frequency for three gene functional categories
  3. Building a semantic word pool for each category via sentence embeddings
  4. Assigning English words to codons by matching biological frequency rank
     to semantic similarity rank — so common codons get common/central words

Gene sources (NCBI Nucleotide, fetched via Biopython Entrez):
  - circadian rhythm:   NM_001035612  (CCA1, CIRCADIAN CLOCK ASSOCIATED 1)
  - photosynthesis:     AY091856
  - stress response:    NM_124370

Embedding model:
  sentence-transformers / all-MiniLM-L6-v2
  Wang et al., "SBERT: Sentence-BERT: Sentence Embeddings using Siamese
  BERT-Networks," EMNLP 2019. https://arxiv.org/abs/1908.10084

Word frequency data:
  wordfreq library — Robyn Speer et al., https://github.com/rspeer/wordfreq
  Provides corpus-derived frequency ranks for English words.

Output:
  codon_word_mapping.json — nested dict:
    { gene_function: { codon: { word, semantic_score, word_frequency, gene_function } } }

Author: Yvonne Wang
"""

from Bio import Entrez, SeqIO
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from wordfreq import top_n_list, word_frequency
import numpy as np
import json

# ── Fetch gene sequences from NCBI ────────────────────────────────────────────

Entrez.email = "your@email.com"  # Required by NCBI API policy

# NCBI accession IDs for the three Arabidopsis gene categories.
# Each category represents a distinct biological signaling domain
# that maps to a distinct poetic register in the installation.
gene_ids = {
    "circadian":       "NM_001035612",  # CCA1 — core circadian clock gene
    "photosynthesis":  "AY091856",      # light-harvesting complex
    "stress_response": "NM_124370"      # defense / wound response
}

sequences = {}
for function, gene_id in gene_ids.items():
    try:
        handle = Entrez.efetch(
            db="nucleotide", id=gene_id,
            rettype="fasta", retmode="text"
        )
        record = SeqIO.read(handle, "fasta")
        sequences[function] = str(record.seq)
        print(f"✓ {function}: {len(record.seq)} bp")
    except Exception as e:
        print(f"✗ {function}: {e}")


# ── Codon frequency analysis ───────────────────────────────────────────────────

def get_codon_frequencies(sequence: str) -> dict:
    """
    Compute the relative usage frequency of every 3-mer (codon) in a DNA sequence.

    Input:  sequence — nucleotide string (A/T/G/C), reading frame starts at position 0
    Output: dict mapping codon (str) → relative frequency (float in [0, 1])

    The frequencies sum to 1.0 across all codons in the sequence.
    High-frequency codons in a gene reflect preferred amino acid usage
    and tRNA abundance — a genuine signal of the gene's expression strategy.

    Artistic intent:
      Frequent codons are later assigned semantically central words (high cosine
      similarity to seed concepts). The most-used biological units become the
      most-used poetic units, preserving the gene's internal hierarchy.
    """
    codons = [sequence[i:i+3]
              for i in range(0, len(sequence) - 2, 3)
              if len(sequence[i:i+3]) == 3]
    freq  = Counter(codons)
    total = sum(freq.values())
    return {c: n / total for c, n in freq.items()}


codon_freqs = {}
for function, seq in sequences.items():
    codon_freqs[function] = get_codon_frequencies(seq)


# ── Semantic word pool construction ───────────────────────────────────────────

# Sentence embedding model used to find semantically similar English words.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Hand-curated seed concepts for each gene function.
# These are not scientific terms but poetic metaphors chosen to anchor
# the semantic field of each biological category.
# The embedding model then radiates outward to find related vocabulary
# from the top 10,000 most common English words.
seed_concepts = {
    "circadian": [
        "cycle", "rhythm", "night", "dawn",
        "return", "pulse", "sleep", "wake"
    ],
    "photosynthesis": [
        "light", "transform", "absorb", "green",
        "energy", "leaf", "radiate", "convert"
    ],
    "stress_response": [
        "boundary", "resist", "harden", "seal",
        "contract", "defend", "pressure", "threshold"
    ]
}

# Words excluded from the candidate pool: too common, too vague, or
# would disrupt the poetic register if they appeared frequently.
stopwords = {
    'something', 'given', 'during', 'home',
    'answer', 'piece', 'cover', 'thing',
    'make', 'come', 'take', 'get', 'just',
    'also', 'back', 'even', 'well', 'still'
}

# Candidate word pool: top 10,000 English words, filtered to alpha-only, len > 2
candidates = [w for w in top_n_list('en', 10000)
              if w.isalpha() and len(w) > 2
              and w not in stopwords]

print(f"\nEncoding {len(candidates)} candidate words...")
candidate_embeddings = model.encode(candidates, show_progress_bar=True)


def build_semantic_pool(seeds: list, candidates: list,
                        candidate_embeddings, top_k: int = 80) -> list:
    """
    Find the top_k English words most semantically related to a set of seed concepts.

    Input:
      seeds               — list of seed concept strings (e.g. ["cycle", "rhythm", …])
      candidates          — list of candidate English words
      candidate_embeddings — pre-computed sentence embeddings for all candidates
      top_k               — number of words to return (default 80)

    Output:
      Sorted list of (word, semantic_score, word_frequency) tuples,
      ordered by descending corpus frequency (most common words first).

    Method:
      Each seed is encoded individually; its cosine similarities to all candidate
      embeddings are accumulated. The top_k candidates by total score form the pool.
      The pool is then re-sorted by word_frequency so that common words are
      mapped to frequent codons — aligning linguistic and biological salience.

    Artistic intent:
      The seed words define the emotional and conceptual register of each
      gene function. The embedding model extends that register to a broader
      vocabulary without manual curation, grounding the poetic vocabulary
      in the latent geometry of language itself.
    """
    all_scores = np.zeros(len(candidates))
    for seed in seeds:
        seed_emb    = model.encode([seed])
        sims        = cosine_similarity(seed_emb, candidate_embeddings)[0]
        all_scores += sims

    top_indices = np.argsort(all_scores)[::-1][:top_k]
    pool = [(candidates[i], all_scores[i]) for i in top_indices]

    return sorted(
        [(w, s, word_frequency(w, 'en')) for w, s in pool],
        key=lambda x: x[2], reverse=True
    )


semantic_pools = {}
for function, seeds in seed_concepts.items():
    semantic_pools[function] = build_semantic_pool(
        seeds, candidates, candidate_embeddings)
    print(f"✓ {function} word pool built ({len(semantic_pools[function])} words)")


# ── Merge into final mapping table ────────────────────────────────────────────

def build_mapping_table(codon_freqs: dict, semantic_pools: dict) -> dict:
    """
    Assign one English word to each codon by rank-matching biological frequency
    to semantic similarity.

    Input:
      codon_freqs    — dict: gene_function → {codon: relative_freq}
      semantic_pools — dict: gene_function → [(word, sem_score, word_freq), …]

    Output:
      Nested dict:
        { gene_function: { codon: { word, semantic_score, word_frequency, gene_function } } }

    Mapping logic:
      For each gene function, codons are sorted by descending frequency.
      The semantic pool is already sorted by descending word frequency.
      The i-th most frequent codon receives the i-th most frequent / central word.

      This creates an isomorphism: biological abundance ↔ linguistic centrality.
      The most-expressed codon in a gene maps to the most culturally common word
      in the corresponding semantic field — e.g. the most frequent circadian codon
      might map to "night" rather than "vespertine."

    Artistic intent:
      The mapping is not arbitrary. It encodes a theory: that the patterns of
      preference embedded in a genome rhyme with the patterns of preference
      embedded in language. Both systems select for efficiency and expressibility.
    """
    mapping = {}
    for function in codon_freqs:
        sorted_codons = sorted(
            codon_freqs[function].items(),
            key=lambda x: x[1], reverse=True)
        pool = semantic_pools[function]
        n    = min(len(sorted_codons), len(pool))
        mapping[function] = {}
        for i, (codon, _) in enumerate(sorted_codons[:n]):
            word, sem_score, freq = pool[i]
            mapping[function][codon] = {
                "word":            word,
                "semantic_score":  round(float(sem_score), 4),
                "word_frequency":  float(freq),
                "gene_function":   function
            }
    return mapping


mapping_table = build_mapping_table(codon_freqs, semantic_pools)

with open('codon_word_mapping.json', 'w') as f:
    json.dump(mapping_table, f, indent=2)

print("\n✓ codon_word_mapping.json saved")
