# Circadian projection-head dataset

## Purpose

This dataset is designed to fine-tune HyenaDNA's A/C/G/T projection head on a
small collection of *Arabidopsis thaliana* circadian-clock genes. It also keeps
two independent test cases outside model selection:

- **ELF4**, a short circadian gene from the Evening Complex, tests whether
  adaptation transfers to an unseen member of the clock system.
- **ECT2**, a non-circadian gene, is a negative control for determining whether
  an improvement is circadian-specific or merely reflects general adaptation to
  *Arabidopsis* sequence composition.

The intended comparison is therefore not just whether fine-tuning lowers loss,
but whether it lowers loss more strongly on held-out ELF4 than on ECT2.

## Directory layout

```text
training/
├── data/
│   ├── circadian/          # Training records; validation tails come from these
│   ├── circadian-held-out/ # Circadian tests, never used for model selection
│   └── non-circadian/      # Non-circadian test controls
├── heads/                  # Fitted projection-head checkpoints
├── runs/                   # Training logs and metrics
└── study/
    ├── models/             # Cross-model comparison table, CSV, and figure
    └── xent/               # Per-model cross-entropy figures
```

The FASTA files remain separate. The trainer constructs windows within each
record and never joins the end of one gene to the beginning of another.

## Training genes

The training set covers morning, daytime, and evening clock components, as well
as light-input and post-translational regulation. This is more representative of
the oscillator than training on CCA1 alone.

| Gene | AGI locus | Length | Reason for inclusion | NCBI links |
|---|---|---:|---|---|
| **CCA1** | AT2G46830 | 3,325 bp | Morning MYB transcription factor and the original gene used by the project; anchors the dataset. | [Gene](https://www.ncbi.nlm.nih.gov/gene/819296) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=19245591&seq_stop=19248915&strand=1&rettype=fasta&retmode=text) |
| **LHY** | AT1G01060 | 4,507 bp | CCA1's close morning-clock partner; including it teaches the head patterns shared by the related CCA1/LHY arm of the oscillator. | [Gene](https://www.ncbi.nlm.nih.gov/gene/839341) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003070.9&seq_start=33365&seq_stop=37871&strand=2&rettype=fasta&retmode=text) |
| **PRR9** | AT2G46790 | 2,573 bp | Morning pseudo-response regulator; part of the temporal PRR sequence and a repressor of CCA1/LHY. | [Gene](https://www.ncbi.nlm.nih.gov/gene/819292) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=19232607&seq_stop=19235179&strand=1&rettype=fasta&retmode=text) |
| **PRR7** | AT5G02810 | 4,350 bp | Later-morning/daytime pseudo-response regulator; complements PRR9 and represses CCA1/LHY. | [Gene](https://www.ncbi.nlm.nih.gov/gene/831793) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=637681&seq_stop=642030&strand=2&rettype=fasta&retmode=text) |
| **PRR5** | AT5G24470 | 2,343 bp | Afternoon pseudo-response regulator; extends coverage across the clock's daily PRR wave. | [Gene](https://www.ncbi.nlm.nih.gov/gene/832518) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=8356204&seq_stop=8358546&strand=2&rettype=fasta&retmode=text) |
| **TOC1 / PRR1** | AT5G61380 | 3,588 bp | Central evening pseudo-response regulator and a core counterpart to morning CCA1/LHY activity. | [Gene](https://www.ncbi.nlm.nih.gov/gene/836259) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=24674963&seq_stop=24678550&strand=1&rettype=fasta&retmode=text) |
| **ELF3** | AT2G25930 | 4,381 bp | Scaffold and essential component of the Evening Complex. | [Gene](https://www.ncbi.nlm.nih.gov/gene/817134) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=11058944&seq_stop=11063324&strand=1&rettype=fasta&retmode=text) |
| **LUX / PCL1** | AT3G46640 | 3,918 bp | DNA-binding component of the Evening Complex and a direct transcriptional clock regulator. | [Gene](https://www.ncbi.nlm.nih.gov/gene/823817) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003074.8&seq_start=17183042&seq_stop=17186959&strand=1&rettype=fasta&retmode=text) |
| **GI** | AT1G22770 | 6,040 bp | Clock-associated regulator linking circadian timing, light signalling, and photoperiodic flowering. | [Gene](https://www.ncbi.nlm.nih.gov/gene/838883) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003070.9&seq_start=8061751&seq_stop=8067790&strand=1&rettype=fasta&retmode=text) |
| **ZTL** | AT5G57360 | 3,271 bp | Clock-associated F-box photoreceptor that controls TOC1 stability and adds post-translational regulation to the set. | [Gene](https://www.ncbi.nlm.nih.gov/gene/835842) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=23241320&seq_stop=23244590&strand=1&rettype=fasta&retmode=text) |

## Held-out circadian test: ELF4

| Gene | AGI locus | Length | NCBI links |
|---|---|---:|---|
| **ELF4** | AT2G40080 | 660 bp | [Gene](https://www.ncbi.nlm.nih.gov/gene/818596) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=16734294&seq_stop=16734953&strand=2&rettype=fasta&retmode=text) |

ELF4 is an Evening Complex component required for robust circadian rhythms. At
only 660 bp, it cannot be divided into separate training and validation regions
with the default 512-base window: one 512-base validation window would leave
only 148 bases for training. Holding it out completely avoids that awkward
special case and gives every training gene the same split semantics.

ELF4 is evaluated only after the best checkpoint has been selected. Its 659
predictable bases can all be scored, because test evaluation is windowed and
does not require separate training and validation portions. This makes the short
sequence useful rather than discarding it or changing the context size for the
whole experiment.

Compared with holding out LHY, holding out ELF4 is also a somewhat harder
transfer test: LHY closely resembles the training anchor CCA1, whereas ELF4 is a
distinct, compact component of the evening clock machinery. Its short length
does mean its estimate will have greater sampling uncertainty than a longer
held-out sequence.

## Non-circadian test: ECT2

| Gene | AGI locus | Current sequence | Length | NCBI links |
|---|---|---|---:|---|
| **ECT2** | AT3G13460 | Genomic gene region `NC_003074.8:c4388592-4384563` | 4,030 bp | [Gene](https://www.ncbi.nlm.nih.gov/gene/820548) · [FASTA](https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003074.8&seq_start=4384563&seq_stop=4388592&strand=2&rettype=fasta&retmode=text) |

ECT2 is not a core circadian-clock component. It is used as a negative control:
if fine-tuning improves ELF4 substantially more than ECT2, that supports a claim
of clock-family adaptation. If both improve similarly, the head may instead be
learning broad *Arabidopsis* nucleotide composition or other shared properties.

ECT2 is stored as its complete chromosome-3 genomic gene region, including
introns, rather than as a spliced mRNA. This matches the sequence type used for
the circadian records and removes the earlier genomic-versus-mRNA confound. A
stronger future control set would still include several additional
length-matched non-circadian genomic genes.

## Sequence source and orientation

The circadian records were obtained from NCBI reference chromosomes as complete
gene intervals. They are stored in transcriptional 5′→3′ orientation. Minus-
strand records were downloaded with NCBI's reverse-complement option and contain
`:c` in their FASTA headers:

- LHY
- PRR7
- PRR5
- ELF4
- ECT2

They must not be reversed or reverse-complemented again. A plain Python reversal
such as `sequence[::-1]` would not be biologically correct in any case because a
strand conversion requires both reversal and nucleotide complementation.

The reference accessions represented here are `NC_003070.9`, `NC_003071.7`,
`NC_003074.8`, and `NC_003076.8`.

## Training and testing protocol

The intended invocation is:

```bash
uv run python -m training.train_projection_head \
  --fasta-files 'training/data/circadian/*.fasta' \
  --fasta-test-files 'training/data/circadian-held-out/*.fasta' \
  --fasta-test-files 'training/data/non-circadian/*.fasta' \
  --device auto
```

Unless `--output` is supplied, the fitted head is saved under `training/heads/`
with the HyenaDNA model configuration in its filename. To train the same dataset
across all supported HyenaDNA sizes sequentially, with one log per run, use:

```bash
training/train_hyenadna_models.sh
```

The heads are written to `training/heads/` and clean, progress-bar-free logs to
`training/runs/`.
The trainer's `--log-file` option mirrors its ordinary output and uncaught errors
to a file while keeping the transient `tqdm` display on the terminal.
Each completed batch run also writes structured `*.metrics.json` data to
`training/runs/`.
After training all models, generate the cross-model table and plot with:

```bash
uv run python -m training.compare_hyenadna_models
```

This produces `training/study/models/hyenadna_model_metrics.md`, a more detailed
CSV alongside it, and
`training/study/models/hyenadna_model_comparison.png`.

Running `uv run python -m training.xent` without `--projection-head` then
discovers those heads and writes one comparison figure per model under
`training/study/xent/`.

For each sufficiently long training gene, a contiguous tail is reserved for
validation and early stopping. Test files are kept completely outside training,
validation, and checkpoint selection. After the best validation checkpoint is
restored, each test file is evaluated independently against both the restricted
pretrained A/C/G/T head and the fine-tuned head.

Validation scores every nucleotide after the first base of each validation tail
exactly once. Overlapping windows are used only to preserve left context; they do
not cause a nucleotide to be counted repeatedly. The combined validation loss is
therefore weighted by actual nucleotide count rather than by the number of
overlapping windows a gene happens to produce.

This separation supports three different interpretations:

1. **Training loss** measures fitting to the selected circadian genes, including
   the related CCA1/LHY morning-clock pair.
2. **Validation loss** measures within-gene generalisation to unseen tails of
   genes represented during training.
3. **Test loss** measures transfer to an entirely held-out circadian gene (ELF4)
   and specificity relative to a non-circadian control (ECT2).

## Reproducibility notes

- Sequence lengths above exclude FASTA headers and line breaks.
- All current sequence bodies contain only A, C, G, and T.
- The direct FASTA links encode the exact reference accession, coordinates, and
  strand used for the local files.
- The seven known HyenaDNA repositories are pinned to the commits recorded in
  `training/hyenadna_models.py`. Use `--model-revision COMMIT_OR_TAG` to override
  the relevant pin when deliberately testing another revision.

## Download the dataset

Run the following from the repository root. `mkdir -p` creates the dataset
directories when needed and succeeds harmlessly when they already exist. Each
`curl` request fails on an HTTP error, follows redirects, and retries transient
network failures.

```bash
mkdir -p training/data/circadian training/data/circadian-held-out \
  training/data/non-circadian

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=19245591&seq_stop=19248915&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.CCA1.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=11058944&seq_stop=11063324&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.ELF3.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003070.9&seq_start=8061751&seq_stop=8067790&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.GI.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003070.9&seq_start=33365&seq_stop=37871&strand=2&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.LHY.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003074.8&seq_start=17183042&seq_stop=17186959&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.LUX.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=8356204&seq_stop=8358546&strand=2&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.PRR5.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=637681&seq_stop=642030&strand=2&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.PRR7.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=19232607&seq_stop=19235179&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.PRR9.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=24674963&seq_stop=24678550&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.TOC1.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003076.8&seq_start=23241320&seq_stop=23244590&strand=1&rettype=fasta&retmode=text' \
  --output training/data/circadian/arabidopsis-thaliana.ZTL.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003071.7&seq_start=16734294&seq_stop=16734953&strand=2&rettype=fasta&retmode=text' \
  --output training/data/circadian-held-out/arabidopsis-thaliana.ELF4.fasta

curl --fail --location --retry 5 \
  'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_003074.8&seq_start=4384563&seq_stop=4388592&strand=2&rettype=fasta&retmode=text' \
  --output training/data/non-circadian/arabidopsis-thaliana.ECT2.fasta
```

## AI attribution

The training and summaries documented here were produced using GPT-5.6-Sol
(medium reasoning effort).
