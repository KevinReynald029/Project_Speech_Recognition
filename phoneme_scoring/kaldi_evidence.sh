#!/usr/bin/env bash
# Native-English nnet3 TDNN (frame factor 1), matching official GOP recipe.
# All intermediates belong to one private TemporaryDirectory; the recipe is read-only.
set -euo pipefail
root=$1
recipe=$2
model=$3
lang=$4
extractor=$5
cd "$recipe"
# Upstream path setup does not promise nounset compatibility.
set +u
. ./path.sh
set -u
mkdir -p "$root/ali/log"
steps/make_mfcc.sh --nj 1 --mfcc-config conf/mfcc_hires.conf --cmd run.pl \
  "$root/data" "$root/mfcc_log" "$root/mfcc"
steps/compute_cmvn_stats.sh "$root/data" "$root/cmvn_log" "$root/mfcc"
utils/fix_data_dir.sh "$root/data"
steps/online/nnet2/extract_ivectors_online.sh --cmd run.pl --nj 1 \
  "$root/data" "$extractor" "$root/ivectors"
steps/nnet3/compute_output.sh --cmd run.pl --nj 1 --apply-exp false \
  --frame-subsampling-factor 1 --online-ivector-dir "$root/ivectors" \
  "$root/data" "$model" "$root/probs"
compile-train-graphs-without-lexicon \
  --read-disambig-syms="$lang/phones/disambig.int" "$model/tree" "$model/final.mdl" \
  "ark,t:$root/text.int" "ark,t:$root/text-phone.int" "ark:|gzip -c > $root/ali/fsts.1.gz"
echo 1 > "$root/ali/num_jobs"
steps/align_mapped.sh --cmd run.pl --nj 1 --graphs "$root/ali" \
  "$root/data" "$root/probs" "$lang" "$model" "$root/ali"
ali-to-phones --write-lengths=true "$model/final.mdl" \
  "ark:gunzip -c $root/ali/ali.1.gz|" "ark,t:$root/lengths.txt"
copy-matrix "ark:$root/probs/output.1.ark" "ark,t:$root/posteriors.txt"
show-transitions "$lang/phones.txt" "$model/final.mdl" > "$root/transitions.txt"
