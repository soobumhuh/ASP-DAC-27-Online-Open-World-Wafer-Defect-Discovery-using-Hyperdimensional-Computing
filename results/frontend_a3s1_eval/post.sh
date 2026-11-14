# after the A3 nested CV: native-grid evaluation + paired bootstrap vs best_clean_adc and vs the control
cd /workspace/wafer-hdc; E=results/frontend_a3s1_eval
until [ -f results/frontend_a3s1/cv_clean.md ]; do sleep 120; done
uv run python scripts/native_eval.py --config configs/frontend_a3_s1.yaml --cases results/frontend_a3s1/cv_clean_cases.pkl \
  --stage merged --workers 12 --out $E/native_frontend_a3s1.json --save-cases $E/native_cases_frontend_a3s1.pkl > $E/native_frontend_a3s1.log 2>&1
R="best=$E/native_cases_clean_adcrim.pkl:merged foc=$E/native_cases_clean_adc_foc.pkl:merged a3s1=$E/native_cases_frontend_a3s1.pkl:merged"
uv run python scripts/bootstrap_ci.py --runs $R --ref best --n 1000 --out $E/boot_vs_best.json > $E/boot_vs_best.log 2>&1
uv run python scripts/bootstrap_ci.py --runs $R --ref foc --n 1000 --out $E/boot_vs_foc.json > $E/boot_vs_foc.log 2>&1
echo A3_POST_DONE >> $E/post.log
