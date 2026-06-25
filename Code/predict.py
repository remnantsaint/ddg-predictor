import argparse
import torch
import esm

from model.model import Predictor


AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "polarity": 8.1, "charge": 0.0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "polarity": 10.5, "charge": 1.0},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "polarity": 11.6, "charge": 0.0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "polarity": 13.0, "charge": -1.0},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "polarity": 5.5, "charge": 0.0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "polarity": 10.5, "charge": 0.0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "polarity": 12.3, "charge": -1.0},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "polarity": 9.0, "charge": 0.0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "polarity": 10.4, "charge": 0.5},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "polarity": 5.2, "charge": 0.0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "polarity": 4.9, "charge": 0.0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "polarity": 11.3, "charge": 1.0},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "polarity": 5.7, "charge": 0.0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "polarity": 5.2, "charge": 0.0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "polarity": 8.0, "charge": 0.0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "polarity": 9.2, "charge": 0.0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "polarity": 8.6, "charge": 0.0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "polarity": 5.4, "charge": 0.0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "polarity": 6.2, "charge": 0.0},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "polarity": 5.9, "charge": 0.0},
}

AA_FLAGS = {
    aa: {
        "aromatic": float(aa in {"F", "W", "Y", "H"}),
        "polar": float(aa in {"S", "T", "N", "Q", "C", "Y", "H", "D", "E", "K", "R"}),
        "aliphatic": float(aa in {"A", "V", "I", "L", "M"}),
        "glycine": float(aa == "G"),
        "proline": float(aa == "P"),
        "cysteine": float(aa == "C"),
    }
    for aa in AA_PROPERTIES
}

SCALAR_KEYS = ["hydrophobicity", "volume", "polarity", "charge"]
FLAG_KEYS = ["aromatic", "polar", "aliphatic", "glycine", "proline", "cysteine"]
AA_FEATURE_DIM = len(SCALAR_KEYS) + len(FLAG_KEYS)


_NORM_MEANS = {
    k: sum(p[k] for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)
    for k in SCALAR_KEYS
}
_NORM_STDS = {
    k: (sum((p[k] - _NORM_MEANS[k]) ** 2 for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)) ** 0.5
    for k in SCALAR_KEYS
}


def pair_physchem_features(wt_aa, mt_aa):
    wt = wt_aa.upper()
    mt = mt_aa.upper()
    features = []
    for k in SCALAR_KEYS:
        wt_norm = (AA_PROPERTIES[wt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        mt_norm = (AA_PROPERTIES[mt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        features.append(mt_norm - wt_norm)
    for k in FLAG_KEYS:
        features.append(AA_FLAGS[mt][k] - AA_FLAGS[wt][k])
    return torch.tensor(features, dtype=torch.float32)


def read_fasta(file_path):
    sequences = {}
    current_id = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = ""
            elif current_id:
                sequences[current_id] += line
    return sequences


def create_mutant(wt_seq, pos, wt_aa, mt_aa):
    idx = pos - 1
    if idx < 0 or idx >= len(wt_seq):
        raise ValueError(f"Position {pos} out of range (1-{len(wt_seq)})")
    if wt_seq[idx] != wt_aa:
        raise ValueError(
            f"WT mismatch: expected {wt_aa}, got {wt_seq[idx]} at position {pos}"
        )
    return wt_seq[:idx] + mt_aa + wt_seq[idx + 1:]


def extract_embedding(model, batch_converter, sequence, device):
    data = [(sequence, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_repr = results["representations"][33].squeeze(0)
        embedding = token_repr[1: len(sequence) + 1, :].cpu()
    return embedding


def load_esm2_model(device):
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def load_predictor(model_path, device):
    model = Predictor().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(pid, pos, wt_aa, mt_aa, fasta_path, model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = read_fasta(fasta_path)
    if pid not in sequences:
        raise KeyError(f"Protein {pid} not found in {fasta_path}")
    wt_seq = sequences[pid]

    mutant_seq = create_mutant(wt_seq, pos, wt_aa, mt_aa)

    esm_model, batch_converter = load_esm2_model(device)
    wt_emb = extract_embedding(esm_model, batch_converter, wt_seq, device)
    vt_emb = extract_embedding(esm_model, batch_converter, mutant_seq, device)

    wt_tensor = wt_emb.unsqueeze(0).float().to(device)
    vt_tensor = vt_emb.unsqueeze(0).float().to(device)
    length = torch.tensor([wt_emb.shape[0]], dtype=torch.long, device=device)

    aa_features = pair_physchem_features(wt_aa, mt_aa).unsqueeze(0).to(device)

    model = load_predictor(model_path, device)
    with torch.no_grad():
        ddg = model(wt_tensor, vt_tensor, lengths=length, aa_features=aa_features).item()

    return ddg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", required=True, help="protein ID in FASTA")
    parser.add_argument("--pos", type=int, required=True, help="mutation position (1-based)")
    parser.add_argument("--wt", required=True, help="wildtype amino acid")
    parser.add_argument("--mt", required=True, help="mutant amino acid")
    parser.add_argument("--fasta", required=True, help="FASTA file containing protein sequences")
    parser.add_argument("--model", default="model/model.pt", help="model weights path")
    args = parser.parse_args()

    result = predict(
        pid=args.pid, pos=args.pos, wt_aa=args.wt, mt_aa=args.mt,
        fasta_path=args.fasta, model_path=args.model,
    )
    print(f"{result:.4f}")
