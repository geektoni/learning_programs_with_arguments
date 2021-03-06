#!/bin/bash

execute_model() {
  python validate_quicksorting.py --output-dir ../validations_paper --op $2 --load-path $1 --save-results
}

operations=("PARTITION_UPDATE" "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE" "QUICKSORT")

for op in ${operations[@]};
do
    ((j=j%1)); ((j++==0)) && wait
    for f in `ls ../models_paper/*.pth`;
    do
      echo "[*] Executing " ${op} ${f}
      execute_model ${f} ${op}
    done
done

