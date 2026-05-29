#!/usr/bin/env python3
"""Fine-tune only the HyenaDNA projection head for CCA1 next-nucleotide prediction."""

import argparse
import os
import random
import re
import sys
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModel, AutoTokenizer

PROJECTION_SAVE_PATH = "projection_head_finetuned.pt"
DEFAULT_MODEL_NAME = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 1
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 40
DEFAULT_LR = 1e-3
DEFAULT_PATIENCE = 5
NUCLEOTIDES = "ACGT"
CLASS_TO_NUC = list(NUCLEOTIDES)
NUC_TO_CLASS = {n: i for i, n in enumerate(NUCLEOTIDES)}


class SequenceWindowDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.input_ids[index]


def parse_fasta(raw: str) -> str:
    """Parse a raw FASTA string or direct nucleotide string into ACGT-only sequence."""
    raw = raw.strip()
    if raw.startswith(">"):
        lines = [line.strip() for line in raw.splitlines() if line and not line.startswith(">")]
        raw = "".join(lines)
    cleaned = re.sub(r"[^ACGTacgt]", "", raw)
    cleaned = cleaned.upper()
    if len(cleaned) == 0:
        raise ValueError("No valid ACGT characters found in FASTA input.")
    return cleaned


def build_windows(sequence: str, window_size: int, stride: int) -> List[str]:
    """Produce a sliding-window list of raw nucleotide strings for tokenizer input."""
    if len(sequence) < window_size:
        raise ValueError(f"Sequence length {len(sequence)} is shorter than window_size {window_size}.")
    windows = [sequence[i : i + window_size] for i in range(0, len(sequence) - window_size + 1, stride)]
    return windows


def load_sequence(args: argparse.Namespace) -> str:
    if args.fasta_file is None and args.fasta_string is None:
        raise ValueError("Provide either --fasta-file or --fasta-string.")
    if args.fasta_file is not None:
        if args.fasta_file == "-":
            raw = sys.stdin.read()
        else:
            with open(args.fasta_file, "r", encoding="utf-8") as handle:
                raw = handle.read()
    else:
        raw = args.fasta_string
    return parse_fasta(raw)


def prepare_dataset(sequence: str, tokenizer: AutoTokenizer, window_size: int, stride: int) -> SequenceWindowDataset:
    windows = build_windows(sequence, window_size, stride)
    # HyenaDNA input is character-level, so each window is tokenized as a raw nucleotide string.
    encoded = tokenizer(windows, padding=True, return_tensors="pt", add_special_tokens=True)
    assert "input_ids" in encoded, "Tokenizer failed to return input IDs."
    return SequenceWindowDataset(encoded["input_ids"])


def get_label_id_map(tokenizer: AutoTokenizer) -> dict:
    """Build mapping from tokenizer nucleotide token IDs to class indices."""
    ids = [tokenizer(nuc, add_special_tokens=False)["input_ids"][0] for nuc in NUCLEOTIDES]
    return {token_id: class_idx for class_idx, token_id in enumerate(ids)}


def remap_labels(labels: torch.Tensor, label_map: dict, ignore_index: int = -100) -> torch.Tensor:
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


def logits_to_distribution(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    masked_probs = probs * mask.unsqueeze(-1).float()
    counts = masked_probs.sum(dim=(0, 1))
    total = mask.sum().item()
    return counts / total if total > 0 else torch.zeros(4, device=logits.device)


def generate_sample(model, projection, tokenizer, seed: str, length: int, context_size: int, device: torch.device) -> str:
    model.eval()
    generated = seed
    for _ in range(length):
        context = generated[-context_size:]
        input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            hidden = model(input_ids)[0]
            logits = projection(hidden[0, -1, :])
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1).item()
        generated += CLASS_TO_NUC[idx]
    return generated


def evaluate(model, projection, dataloader, label_map: dict, device: torch.device) -> tuple[float, torch.Tensor]:
    model.eval()
    projection.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    total_probs = torch.zeros(4, device=device)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            hidden = outputs[0]
            logits = projection(hidden[:, :-1, :])
            labels = remap_labels(batch[:, 1:].clone(), label_map)
            mask = labels != -100
            loss = criterion(logits.view(-1, 4), labels.view(-1))
            total_loss += loss.item() * mask.sum().item()
            total_probs += logits_to_distribution(logits, mask)
            total_tokens += mask.sum().item()
    avg_loss = total_loss / total_tokens if total_tokens else float("inf")
    avg_probs = total_probs / total_tokens if total_tokens else torch.zeros(4, device=device)
    return avg_loss, avg_probs


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune HyenaDNA projection head on CCA1 nucleotide sequence.")
    parser.add_argument("--fasta-file", type=str, help="Path to a FASTA file containing the CCA1 sequence.")
    parser.add_argument("--fasta-string", type=str, help="Raw FASTA contents or raw A/C/G/T string.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                        help="Sliding window size for HyenaDNA tokenization (default: 512).")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Sliding window stride over the training sequence.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Learning rate for the projection head optimizer.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="Early stopping patience on validation loss.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Hugging Face model name or path for HyenaDNA.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device to use, e.g. cuda or cpu.")
    args = parser.parse_args()

    if args.fasta_file is None and args.fasta_string is None:
        parser.error("Provide either --fasta-file or --fasta-string.")

    sequence = parse_fasta(args.fasta_string if args.fasta_string is not None else open(args.fasta_file, "r", encoding="utf-8").read())
    print(f"Loaded sequence length: {len(sequence)}")
    actual_dist = compute_nucleotide_distribution(sequence)
    print("Actual CCA1 nucleotide distribution:")
    print("  " + ", ".join([f"{n}:{actual_dist[n]*100:.1f}%" for n in NUCLEOTIDES]))

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    label_map = get_label_id_map(tokenizer)
    data = prepare_dataset(sequence, tokenizer, args.window_size, args.stride)
    print(f"Prepared {len(data)} sliding windows of size {args.window_size}.")

    val_size = max(1, int(len(data) * 0.1))
    train_size = len(data) - val_size
    train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    projection = torch.nn.Linear(model.config.hidden_size, 4, bias=False).to(device)
    torch.nn.init.xavier_uniform_(projection.weight)
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        projection.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        for batch in train_loader:
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

        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens else float("inf")
        val_loss, val_probs = evaluate(model, projection, val_loader, label_map, device)
        val_dist = {nuc: float(val_probs[idx].cpu().item()) for idx, nuc in enumerate(NUCLEOTIDES)}

        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}")
        print("  Validation nucleotide distribution: " + ", ".join([f"{n}:{val_dist[n]*100:.1f}%" for n in NUCLEOTIDES]))

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in projection.state_dict().items()}
            patience = 0
            print("  ✓ New best validation loss; saving snapshot in memory.")
        else:
            patience += 1
            print(f"  Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        torch.save(best_state, PROJECTION_SAVE_PATH)
        print(f"Saved fine-tuned projection head to {PROJECTION_SAVE_PATH}")
    else:
        torch.save(projection.state_dict(), PROJECTION_SAVE_PATH)
        print(f"Saved final projection head to {PROJECTION_SAVE_PATH}")

    sample_seed = sequence[: min(args.window_size, len(sequence))]
    sample = generate_sample(model, projection, tokenizer, sample_seed, length=100, context_size=args.window_size, device=device)
    sample_counts = compute_nucleotide_distribution(sample[len(sample_seed) :])
    print("Generated sequence sample (100 nt):")
    print(sample[len(sample_seed) :])
    print("Generated nucleotide distribution:")
    print("  " + ", ".join([f"{n}:{sample_counts[n]*100:.1f}%" for n in NUCLEOTIDES]))


if __name__ == "__main__":
    main()
