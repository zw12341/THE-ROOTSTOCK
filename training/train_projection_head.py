#!/usr/bin/env python3
"""Fine-tune the HyenaDNA projection head on one or more FASTA files."""

import argparse
import glob
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__:
    from .hyenadna_models import (
        AUTO_MODEL_REVISION,
        DEFAULT_HYENADNA_MODEL,
        resolve_hyenadna_revision,
    )
else:
    from hyenadna_models import (
        AUTO_MODEL_REVISION,
        DEFAULT_HYENADNA_MODEL,
        resolve_hyenadna_revision,
    )

DEFAULT_MODEL_NAME = DEFAULT_HYENADNA_MODEL
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 40
DEFAULT_LR = 1e-4
DEFAULT_PATIENCE = 8
DEFAULT_LR_PLATEAU_PATIENCE = 2
DEFAULT_LR_REDUCTION_FACTOR = 1.0 / 3.0
DEFAULT_MIN_LR = 1e-6
DEFAULT_VALIDATION_FRACTION = 0.1
NUCLEOTIDES = "ACGT"
CLASS_TO_NUC = list(NUCLEOTIDES)
NUC_TO_CLASS = {n: i for i, n in enumerate(NUCLEOTIDES)}
PROGRESS_STREAM = sys.stderr


class TeeStream:
    """Write ordinary console output to both the terminal and a clean log file."""

    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()


def configure_run_log(path: Path) -> None:
    """Mirror console output to a file while tqdm keeps its original stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log)
    sys.stderr = TeeStream(sys.stderr, log)
    print(f"Logging run output to {path}")


def default_projection_path(model_name: str) -> str:
    """Build a concise checkpoint name from a HyenaDNA model ID or path."""
    model_id = Path(model_name.rstrip("/")).name
    model_data = re.sub(r"^hyenadna-", "", model_id)
    model_data = model_data.replace("-seqlen-", "-")
    model_data = re.sub(r"-hf$", "", model_data)
    return str(
        Path("training/heads") / f"projection_head_hyenadna_{model_data}.pt"
    )


class SequenceWindowDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.input_ids[index]


@dataclass(frozen=True)
class FastaRecord:
    source: Path
    name: str
    sequence: str


def expand_fasta_patterns(patterns: list[str]) -> list[Path]:
    """Expand shell-expanded paths and quoted glob patterns, preserving order."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"FASTA path or pattern matched nothing: {pattern}")
        for path in matches:
            if not path.is_file():
                raise ValueError(f"FASTA input is not a file: {path}")
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def read_fasta_records(path: Path) -> list[FastaRecord]:
    """Read FASTA records without joining separate records together."""
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_lines: list[str] = []

    def finish_record() -> None:
        if header is None:
            return
        sequence = "".join(sequence_lines).upper()
        invalid = sorted(set(sequence) - set(NUCLEOTIDES))
        if invalid:
            raise ValueError(
                f"{path} record {header!r} contains unsupported symbols: "
                f"{', '.join(invalid)}"
            )
        if not sequence:
            raise ValueError(f"{path} record {header!r} has no sequence.")
        records.append(FastaRecord(path, header, sequence))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
            sequence_lines = []
        else:
            if header is None:
                raise ValueError(f"{path} contains sequence data before a FASTA header.")
            sequence_lines.append(re.sub(r"\s", "", line))
    finish_record()
    if not records:
        raise ValueError(f"{path} contains no FASTA records.")
    return records


def load_records(patterns: list[str]) -> tuple[list[Path], list[FastaRecord]]:
    paths = expand_fasta_patterns(patterns)
    records = [record for path in paths for record in read_fasta_records(path)]
    return paths, records


def build_windows(sequence: str, window_size: int, stride: int) -> list[str]:
    """Produce a sliding-window list of raw nucleotide strings for tokenizer input."""
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if stride < 1:
        raise ValueError("stride must be at least 1.")
    if len(sequence) < window_size:
        raise ValueError(
            f"Sequence length {len(sequence)} is shorter than window_size {window_size}."
        )
    windows = [
        sequence[i : i + window_size]
        for i in range(0, len(sequence) - window_size + 1, stride)
    ]
    return windows


def split_training_records(
    records: list[FastaRecord], window_size: int, validation_fraction: float
) -> tuple[list[str], list[str]]:
    """Split each sufficiently long record without creating cross-gene windows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    training_sequences: list[str] = []
    validation_sequences: list[str] = []
    for record in records:
        length = len(record.sequence)
        if length < window_size:
            raise ValueError(
                f"{record.source} record {record.name!r} is {length} bases, shorter "
                f"than --window-size {window_size}."
            )
        validation_size = max(window_size, math.ceil(length * validation_fraction))
        training_size = length - validation_size
        if training_size < window_size:
            # A short record such as ELF4 can still contribute to fitting, but cannot
            # supply a separate full-size validation window.
            training_sequences.append(record.sequence)
            print(
                f"  {record.source} ({length:,} bases): all training; too short for "
                f"a non-overlapping {window_size}-base validation window."
            )
        else:
            training_sequences.append(record.sequence[:training_size])
            validation_sequences.append(record.sequence[training_size:])
            print(
                f"  {record.source} ({length:,} bases): {training_size:,} training, "
                f"{validation_size:,} validation."
            )
    if not validation_sequences:
        raise ValueError(
            "None of the training FASTA records is long enough to provide separate "
            "training and validation windows. Use a smaller --window-size."
        )
    return training_sequences, validation_sequences


def prepare_dataset(
    sequences: list[str], tokenizer: AutoTokenizer, window_size: int, stride: int
) -> SequenceWindowDataset:
    windows = [
        window
        for sequence in sequences
        for window in build_windows(sequence, window_size, stride)
    ]
    # HyenaDNA input is character-level, so each window is tokenized as a raw nucleotide string.
    encoded = tokenizer(
        windows, padding=True, return_tensors="pt", add_special_tokens=True
    )
    assert "input_ids" in encoded, "Tokenizer failed to return input IDs."
    return SequenceWindowDataset(encoded["input_ids"])


def get_label_id_map(tokenizer: AutoTokenizer) -> dict:
    """Build mapping from tokenizer nucleotide token IDs to class indices."""
    ids = [
        tokenizer(nuc, add_special_tokens=False)["input_ids"][0] for nuc in NUCLEOTIDES
    ]
    return {token_id: class_idx for class_idx, token_id in enumerate(ids)}


def make_projection(
    causal_model,
    label_map: dict[int, int],
    initialization: str,
    device: torch.device,
) -> torch.nn.Linear:
    """Create the four-way head, normally from the pretrained LM head rows."""
    projection = torch.nn.Linear(causal_model.config.d_model, 4, bias=False).to(device)
    if initialization == "random":
        torch.nn.init.xavier_uniform_(projection.weight)
        return projection

    output_head = causal_model.get_output_embeddings()
    if output_head is None or output_head.weight.ndim != 2:
        raise RuntimeError("The pretrained model does not expose a linear output head.")
    with torch.no_grad():
        for token_id, class_idx in label_map.items():
            projection.weight[class_idx].copy_(output_head.weight[token_id])
    return projection


def remap_labels(
    labels: torch.Tensor, label_map: dict, ignore_index: int = -100
) -> torch.Tensor:
    mapped = torch.full_like(labels, ignore_index)
    for token_id, class_idx in label_map.items():
        mapped[labels == token_id] = class_idx
    return mapped


def compute_nucleotide_distribution(sequence: str) -> dict:
    counts = {nuc: 0 for nuc in NUCLEOTIDES}
    for char in sequence:
        if char in counts:
            counts[char] += 1
    total = sum(counts.values())
    return {nuc: counts[nuc] / total for nuc in NUCLEOTIDES} if total else counts


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device auto.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available; use --device auto.")
    return torch.device(requested)


def generate_sample(
    model,
    projection,
    tokenizer,
    seed: str,
    length: int,
    context_size: int,
    device: torch.device,
) -> str:
    model.eval()
    generated = seed
    for _ in range(length):
        context = generated[-context_size:]
        input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)
        with torch.no_grad():
            hidden = model(input_ids)[0]
            logits = projection(hidden[0, -1, :])
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
        generated += CLASS_TO_NUC[idx]
    return generated


def evaluate_sequences_once(
    model,
    projection,
    tokenizer,
    sequences: list[str],
    device: torch.device,
    context_size: int,
) -> tuple[float, torch.Tensor]:
    """Score each validation nucleotide once, using overlap only as context."""
    model.eval()
    projection.eval()
    total_loss = 0.0
    total_tokens = 0
    total_probs = torch.zeros(4, device=device)
    scoring_stride = max(1, context_size // 2)
    with torch.inference_mode():
        for sequence in sequences:
            input_ids = torch.tensor(
                tokenizer.convert_tokens_to_ids(list(sequence)),
                dtype=torch.long,
                device=device,
            )
            labels = torch.tensor(
                [NUC_TO_CLASS[base] for base in sequence],
                dtype=torch.long,
                device=device,
            )
            cursor = 1
            while cursor < len(sequence):
                end = (
                    min(len(sequence), context_size)
                    if cursor == 1
                    else min(len(sequence), cursor + scoring_stride)
                )
                start = max(0, end - context_size)
                hidden = model(input_ids[start:end].unsqueeze(0))[0]
                targets = torch.arange(cursor, end, device=device)
                prediction_positions = targets - start - 1
                target_labels = labels.index_select(0, targets)
                logits = projection(
                    hidden[0].index_select(0, prediction_positions)
                )
                total_loss += F.cross_entropy(
                    logits, target_labels, reduction="sum"
                ).item()
                total_probs += F.softmax(logits, dim=-1).sum(dim=0)
                total_tokens += end - cursor
                cursor = end
    avg_loss = total_loss / total_tokens if total_tokens else float("inf")
    avg_probs = (
        total_probs / total_tokens if total_tokens else torch.zeros(4, device=device)
    )
    return avg_loss, avg_probs


def score_test_record(
    record: FastaRecord,
    causal_model,
    projection: torch.nn.Linear,
    tokenizer,
    label_map: dict[int, int],
    device: torch.device,
    context_size: int,
) -> tuple[float, float, int]:
    """Score every test base once with the pretrained and fine-tuned heads."""
    sequence = record.sequence
    if len(sequence) < 2:
        raise ValueError(f"Test record {record.name!r} must contain at least two bases.")
    class_token_ids = [
        token_id
        for token_id, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]
    input_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(list(sequence)), dtype=torch.long, device=device
    )
    nucleotide_ids = torch.tensor(class_token_ids, dtype=torch.long, device=device)
    labels = torch.tensor(
        [NUC_TO_CLASS[base] for base in sequence], dtype=torch.long, device=device
    )
    base_nll = 0.0
    tuned_nll = 0.0
    token_count = 0
    cursor = 1
    scoring_stride = max(1, context_size // 2)
    output_head = causal_model.get_output_embeddings()
    backbone = causal_model.base_model
    with torch.inference_mode():
        while cursor < len(sequence):
            end = (
                min(len(sequence), context_size)
                if cursor == 1
                else min(len(sequence), cursor + scoring_stride)
            )
            start = max(0, end - context_size)
            hidden = backbone(input_ids[start:end].unsqueeze(0))[0]
            targets = torch.arange(cursor, end, device=device)
            prediction_positions = targets - start - 1
            target_labels = labels.index_select(0, targets)
            selected_hidden = hidden[0].index_select(0, prediction_positions)
            base_logits = output_head(selected_hidden).index_select(-1, nucleotide_ids)
            tuned_logits = projection(selected_hidden)
            base_nll += F.cross_entropy(
                base_logits, target_labels, reduction="sum"
            ).item()
            tuned_nll += F.cross_entropy(
                tuned_logits, target_labels, reduction="sum"
            ).item()
            token_count += end - cursor
            cursor = end
    return base_nll, tuned_nll, token_count


def evaluate_test_files(
    paths: list[Path],
    causal_model,
    projection: torch.nn.Linear,
    tokenizer,
    label_map: dict[int, int],
    device: torch.device,
    context_size: int,
) -> list[dict[str, object]]:
    """Report independent file-level tests after model selection is complete."""
    print("\nIndependent test evaluations (not used for training or validation):")
    results: list[dict[str, object]] = []
    for path in paths:
        records = read_fasta_records(path)
        base_nll = 0.0
        tuned_nll = 0.0
        token_count = 0
        for record in records:
            record_base, record_tuned, record_tokens = score_test_record(
                record,
                causal_model,
                projection,
                tokenizer,
                label_map,
                device,
                context_size,
            )
            base_nll += record_base
            tuned_nll += record_tuned
            token_count += record_tokens
        base_loss = base_nll / token_count
        tuned_loss = tuned_nll / token_count
        gene = path.stem.rsplit(".", 1)[-1]
        print(f"  Test file: {path} (gene {gene}; {token_count:,} scored bases)")
        print(
            f"    pretrained: CE={base_loss:.6f}, perplexity={math.exp(base_loss):.4f}"
        )
        print(
            f"    fine-tuned: CE={tuned_loss:.6f}, perplexity={math.exp(tuned_loss):.4f}"
        )
        print(f"    mean CE improvement: {base_loss - tuned_loss:+.6f} nats")
        results.append(
            {
                "file": str(path),
                "gene": gene,
                "scored_bases": token_count,
                "pretrained_ce": base_loss,
                "pretrained_perplexity": math.exp(base_loss),
                "fine_tuned_ce": tuned_loss,
                "fine_tuned_perplexity": math.exp(tuned_loss),
                "ce_improvement": base_loss - tuned_loss,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune a HyenaDNA projection head on a FASTA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta-files",
        nargs="+",
        action="extend",
        required=True,
        metavar="PATH_OR_GLOB",
        help=(
            "Training FASTA paths or glob patterns. The option may be repeated; "
            "quote a glob to have this program expand it."
        ),
    )
    parser.add_argument(
        "--fasta-test-files",
        nargs="+",
        action="extend",
        default=[],
        metavar="PATH_OR_GLOB",
        help=(
            "Independent test FASTA paths or glob patterns, evaluated separately "
            "after training and model selection. The option may be repeated."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Sliding window size for HyenaDNA tokenization.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Sliding window stride over the training sequence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Learning rate for the projection head optimizer.",
    )
    parser.add_argument(
        "--initialization",
        choices=("pretrained", "random"),
        default="pretrained",
        help=(
            "Initialize the A/C/G/T rows from HyenaDNA's pretrained LM head, or "
            "start a new random classifier."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience on validation loss.",
    )
    parser.add_argument(
        "--lr-plateau-patience",
        type=int,
        default=DEFAULT_LR_PLATEAU_PATIENCE,
        help="Flat validation epochs allowed before reducing the learning rate.",
    )
    parser.add_argument(
        "--lr-reduction-factor",
        type=float,
        default=DEFAULT_LR_REDUCTION_FACTOR,
        help="Factor applied to the learning rate after a validation plateau.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=DEFAULT_MIN_LR,
        help="Minimum learning rate used by the plateau scheduler.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction held out as a contiguous validation tail.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or path for HyenaDNA.",
    )
    parser.add_argument(
        "--model-revision",
        default=AUTO_MODEL_REVISION,
        help=(
            "Hugging Face commit, tag, or branch; 'auto' selects the pinned commit "
            "for each known HyenaDNA model."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Projection-head output path. By default, derives the filename from "
            "--model-name (for example, "
            "training/heads/projection_head_hyenadna_tiny-1k.pt)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Mirror training messages and uncaught exceptions to this file. Live "
            "tqdm redraws are deliberately excluded."
        ),
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Write final validation and independent-test metrics as JSON.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="auto",
        help="Compute device; 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    args = parser.parse_args()
    if args.log_file is not None:
        configure_run_log(args.log_file)
    if args.output is None:
        args.output = default_projection_path(args.model_name)

    training_paths, training_records = load_records(args.fasta_files)
    test_paths = (
        expand_fasta_patterns(args.fasta_test_files) if args.fasta_test_files else []
    )
    overlap = {path.resolve() for path in training_paths} & {
        path.resolve() for path in test_paths
    }
    if overlap:
        duplicates = ", ".join(str(path) for path in sorted(overlap))
        raise ValueError(
            f"Training and independent test inputs overlap: {duplicates}"
        )
    print(
        f"Loaded {len(training_records)} training record(s) from "
        f"{len(training_paths)} FASTA file(s)."
    )
    for path in training_paths:
        print(f"  Training file: {path}")
    total_sequence = "".join(record.sequence for record in training_records)
    actual_dist = compute_nucleotide_distribution(total_sequence)
    print(f"Training nucleotide distribution across {len(total_sequence):,} bases:")
    print("  " + ", ".join([f"{n}:{actual_dist[n] * 100:.1f}%" for n in NUCLEOTIDES]))

    device = choose_device(args.device)
    print(f"Using device: {device}")

    model_revision = resolve_hyenadna_revision(args.model_name, args.model_revision)
    print(f"Model revision: {model_revision or 'unpinned repository default'}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=model_revision,
        trust_remote_code=True,
    )
    causal_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=model_revision,
        trust_remote_code=True,
        return_dict=True,
    ).to(device)
    for param in causal_model.parameters():
        param.requires_grad = False
    causal_model.eval()
    model = causal_model.base_model
    model.eval()

    label_map = get_label_id_map(tokenizer)
    print("Per-record non-overlapping training/validation split:")
    train_sequences, val_sequences = split_training_records(
        training_records, args.window_size, args.validation_fraction
    )
    train_data = prepare_dataset(
        train_sequences, tokenizer, args.window_size, args.stride
    )
    validation_targets = sum(len(sequence) - 1 for sequence in val_sequences)
    print(
        f"Prepared {len(train_data)} training windows of size {args.window_size}; "
        f"validation will score {validation_targets:,} nucleotide targets once."
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    projection = make_projection(
        causal_model, label_map, args.initialization, device
    )
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_reduction_factor,
        patience=args.lr_plateau_patience,
        threshold=1e-6,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Epoch zero is a meaningful baseline when initialized from the pretrained
    # A/C/G/T rows. Keeping it as a candidate guarantees that the saved head is
    # never worse on validation than the restricted pretrained head.
    initial_val_loss, initial_val_probs = evaluate_sequences_once(
        model,
        projection,
        tokenizer,
        val_sequences,
        device,
        args.window_size,
    )
    initial_val_dist = {
        nuc: float(initial_val_probs[idx].cpu().item())
        for idx, nuc in enumerate(NUCLEOTIDES)
    }
    print(
        f"Epoch 00 ({args.initialization} initialization): "
        f"val_loss={initial_val_loss:.6f}"
    )
    print(
        "  Validation nucleotide distribution: "
        + ", ".join([f"{n}:{initial_val_dist[n] * 100:.1f}%" for n in NUCLEOTIDES])
    )
    best_val_loss = initial_val_loss
    best_epoch = 0
    best_state = {k: v.cpu().clone() for k, v in projection.state_dict().items()}
    patience = 0
    last_epoch = 0
    final_val_loss = initial_val_loss
    scheduler.step(initial_val_loss)

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        projection.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            file=PROGRESS_STREAM,
            disable=not PROGRESS_STREAM.isatty(),
        )
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            hidden = outputs[0]
            logits = projection(hidden[:, :-1, :])
            labels = remap_labels(batch[:, 1:].clone(), label_map)
            loss = criterion(logits.view(-1, 4), labels.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * (labels != -100).sum().item()
            epoch_tokens += (labels != -100).sum().item()
            progress.set_postfix(
                train_loss=f"{epoch_loss / epoch_tokens:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens else float("inf")
        val_loss, val_probs = evaluate_sequences_once(
            model,
            projection,
            tokenizer,
            val_sequences,
            device,
            args.window_size,
        )
        final_val_loss = val_loss
        val_dist = {
            nuc: float(val_probs[idx].cpu().item())
            for idx, nuc in enumerate(NUCLEOTIDES)
        }

        print(
            f"Epoch {epoch:02d}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}"
        )
        print(
            "  Validation nucleotide distribution: "
            + ", ".join([f"{n}:{val_dist[n] * 100:.1f}%" for n in NUCLEOTIDES])
        )

        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < previous_lr:
            print(
                f"  Learning-rate plateau: reducing LR from {previous_lr:.2e} "
                f"to {current_lr:.2e}."
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                k: v.cpu().clone() for k, v in projection.state_dict().items()
            }
            best_epoch = epoch
            patience = 0
            print("  ✓ New best validation loss; saving snapshot in memory.")
        else:
            patience += 1
            print(f"  Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    projection.load_state_dict(best_state)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(
        f"Saved best projection head from epoch {best_epoch} to {args.output} "
        f"(validation loss {best_val_loss:.6f})"
    )

    sample_seed = training_records[0].sequence[: args.window_size]
    # Generate with the same best checkpoint that was written to disk.
    sample = generate_sample(
        model,
        projection,
        tokenizer,
        sample_seed,
        length=100,
        context_size=args.window_size,
        device=device,
    )
    sample_counts = compute_nucleotide_distribution(sample[len(sample_seed) :])
    print("Generated sequence sample (100 nt):")
    print(sample[len(sample_seed) :])
    print("Generated nucleotide distribution:")
    print("  " + ", ".join([f"{n}:{sample_counts[n] * 100:.1f}%" for n in NUCLEOTIDES]))

    test_results: list[dict[str, object]] = []
    if test_paths:
        test_results = evaluate_test_files(
            test_paths,
            causal_model,
            projection,
            tokenizer,
            label_map,
            device,
            args.window_size,
        )

    if args.metrics_file is not None:
        metrics = {
            "model": args.model_name,
            "model_revision": model_revision,
            "projection_head": str(args.output),
            "training_files": [str(path) for path in training_paths],
            "best_epoch": best_epoch,
            "epochs_completed": last_epoch,
            "initial_validation_ce": initial_val_loss,
            "best_validation_ce": best_val_loss,
            "validation_ce_improvement": initial_val_loss - best_val_loss,
            "final_validation_ce": final_val_loss,
            "validation_scored_bases": validation_targets,
            "test_results": test_results,
        }
        args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_file.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved structured run metrics to {args.metrics_file}")


if __name__ == "__main__":
    main()
