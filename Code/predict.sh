#!/bin/bash
FASTA="${1:-example/data/proteins.fasta}"

while read pid pos wt mt; do
    echo "Processing: $pid $wt$pos$mt"
    python predict.py --pid "$pid" --pos "$pos" --wt "$wt" --mt "$mt" --fasta "$FASTA"
done < 'mutations.txt'
