set -euo pipefail
rc=0
for nb in notebooks/**/*.ipynb; do
  echo "▶ Executing $nb"
  if jupyter nbconvert --to notebook --execute "$nb" \
        --ExecutePreprocessor.timeout=1200 \
        --output /dev/null; then
    echo "✅ OK: $nb"
  else
    echo "❌ FAIL: $nb"
    rc=1
  fi
done
exit $rc
