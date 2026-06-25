# Protein ddG Predictor

Predict the change in protein folding stability (ddG) caused by a single amino acid mutation.

---

## 1. Setup

Python 3.8 or later is required. CUDA is recommended but not required.

```bash
pip install -r requirements.txt
```

The first run will automatically download the ESM2 model (~2.5 GB).

---

## 2. Quick Test

Run the included example to verify everything works:

```bash
python predict.py \
    --pid P00149 \
    --pos 94 \
    --wt E \
    --mt H \
    --fasta example/data/proteins.fasta
```

This predicts the ddG for mutation E94H on protein P00149. Expected output:

```
-0.2645
```

---

## 3. Run on Your Own Data

### Step 1: Prepare a FASTA file

```fasta
>my_protein
MKLRIATIAGLVVLGSGFAVAQTDVIAQRKAILKQM...
>another_protein
MTKAVCVLKGDGPVQGIINF...
```

Save it as `my_proteins.fasta`.

### Step 2: Run prediction

```bash
python predict.py --pid my_protein --pos 20 --wt A --mt G --fasta my_proteins.fasta
```

The result is printed to stdout as a single float (predicted ddG in kcal/mol).

### Step 3 (optional): Batch prediction

Create `mutations.txt` in the same directory as `predict.sh`. Each line has four columns: `pid  pos  wt  mt`. All PIDs must exist in the FASTA file.

```
my_protein 20 A G
my_protein 35 K T
another_protein 15 R A
```

Then run:

```bash
chmod +x predict.sh
./predict.sh my_proteins.fasta
```

---

## Arguments

| Argument | Description |
|----------|-------------|
| `--pid` | Protein ID matching the FASTA header |
| `--pos` | Mutation position (1-based) |
| `--wt` | Wildtype amino acid (single letter) |
| `--mt` | Mutant amino acid (single letter) |
| `--fasta` | Path to the FASTA file |
| `--model` | Path to model weights (default: model/model.pt) |
