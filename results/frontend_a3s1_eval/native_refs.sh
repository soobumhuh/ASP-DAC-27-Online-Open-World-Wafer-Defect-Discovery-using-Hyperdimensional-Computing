cd /workspace/wafer-hdc
for n in clean_adcrim:best_clean_adc clean_adc_foc:best_clean_adc_foc; do d=${n%%:*}; c=${n##*:}
  uv run python scripts/native_eval.py --config configs/$c.yaml --cases results/$d/cv_clean_cases.pkl --stage merged --workers 8 \
    --out results/frontend_a3s1_eval/native_$d.json --save-cases results/frontend_a3s1_eval/native_cases_$d.pkl > results/frontend_a3s1_eval/native_$d.log 2>&1
done
echo REFS_DONE >> results/frontend_a3s1_eval/native_clean_adc_foc.log
