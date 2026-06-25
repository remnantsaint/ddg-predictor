import torch
import esm
from flask import Flask, request, render_template, jsonify

from model import Predictor, AA_FEATURE_DIM

app = Flask(__name__)

MODEL_PATH = "model.pt"

SCALAR_KEYS = ["hydrophobicity", "volume", "polarity", "charge"]
FLAG_KEYS = ["aromatic", "polar", "aliphatic", "glycine", "proline", "cysteine"]

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

_NORM_MEANS = {
    k: sum(p[k] for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)
    for k in SCALAR_KEYS
}
_NORM_STDS = {
    k: (sum((p[k] - _NORM_MEANS[k]) ** 2 for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)) ** 0.5
    for k in SCALAR_KEYS
}


def pair_physchem_features(wt_aa, mt_aa):
    wt, mt = wt_aa.upper(), mt_aa.upper()
    features = []
    for k in SCALAR_KEYS:
        wt_n = (AA_PROPERTIES[wt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        mt_n = (AA_PROPERTIES[mt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        features.append(mt_n - wt_n)
    for k in FLAG_KEYS:
        features.append(AA_FLAGS[mt][k] - AA_FLAGS[wt][k])
    return torch.tensor(features, dtype=torch.float32)


esm_model = None
batch_converter = None
predictor = None
device = None


def load_models():
    global esm_model, batch_converter, predictor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    predictor = Predictor().to(device).eval()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    predictor.load_state_dict(state_dict)


def extract_embedding(sequence):
    data = [(sequence, sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_repr = results["representations"][33].squeeze(0)
        embedding = token_repr[1: len(sequence) + 1, :].cpu()
    return embedding


def do_predict(sequence, pos, wt_aa, mt_aa):
    idx = pos - 1
    if idx < 0 or idx >= len(sequence):
        return None, f"Position {pos} is out of range (1-{len(sequence)})."
    if sequence[idx].upper() != wt_aa.upper():
        return None, f"Wildtype mismatch: position {pos} is {sequence[idx]}, not {wt_aa}."
    mutant_seq = sequence[:idx] + mt_aa + sequence[idx + 1:]

    wt_emb = extract_embedding(sequence)
    vt_emb = extract_embedding(mutant_seq)

    wt_tensor = wt_emb.unsqueeze(0).float().to(device)
    vt_tensor = vt_emb.unsqueeze(0).float().to(device)
    length = torch.tensor([wt_emb.shape[0]], dtype=torch.long, device=device)
    aa_feat = pair_physchem_features(wt_aa, mt_aa).unsqueeze(0).to(device)

    with torch.no_grad():
        ddg = predictor(wt_tensor, vt_tensor, lengths=length, aa_features=aa_feat).item()

    return ddg, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    sequence = data.get("sequence", "").strip().upper()
    pos = data.get("pos")
    wt = data.get("wt", "").strip().upper()
    mt = data.get("mt", "").strip().upper()

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    for c in sequence:
        if c not in valid_aa:
            return jsonify({"error": f"Invalid character in sequence: '{c}'."})

    try:
        pos = int(pos)
    except (TypeError, ValueError):
        return jsonify({"error": "Position must be an integer."})

    if wt not in valid_aa or mt not in valid_aa:
        return jsonify({"error": "Invalid amino acid code."})

    try:
        result, error = do_predict(sequence, pos, wt, mt)
        if error:
            return jsonify({"error": error})
        return jsonify({
            "ddg": round(result, 4),
            "length": len(sequence),
            "mutation": f"{wt}{pos}{mt}",
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    load_models()
    app.run(host="0.0.0.0", port=6006)
