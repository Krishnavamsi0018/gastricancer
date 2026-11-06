#!/usr/bin/env bash
# Fetch the six GEO series matrices used by the pipeline (~430 MB).
# Platform annotation tables (GPL570, GPL6947) and the clinical annotation files
# are downloaded separately from the corresponding GEO platform and series pages.
set -euo pipefail
mkdir -p data

base="https://ftp.ncbi.nlm.nih.gov/geo/series"

fetch () {           # $1 = accession
  local acc="$1"
  local stub="${acc:0:$(( ${#acc} - 3 ))}nnn"
  local url="${base}/${stub}/${acc}/matrix/${acc}_series_matrix.txt.gz"
  echo "==> ${acc}"
  curl -fL --retry 3 -o "data/${acc}.txt.gz" "${url}"
}

for acc in GSE66229 GSE54129 GSE13911 GSE79973 GSE84437 GSE15459; do
  fetch "${acc}"
done

echo
echo "Series matrices written to data/."
echo "Still required in data/ (see README):"
echo "  GPL570_platform.txt  GPL6947_platform.txt"
echo "  GSE84437_meta.txt  GSE15459_meta.txt  GSE62254_meta.txt  GSE15459_outcome.xls"
