#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --style)
    STYLE="$2"; shift 2;;
  --model_dir)
    MODEL_DIR="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

DATA_DIR=data/CNN_NYT/processed
MODEL=$MODEL_DIR/checkpoint_best.pt

fairseq-generate $DATA_DIR \
        --path $MODEL \
        --user-dir mass \
        --task translation_mix \
        --model_lang_pairs src-tgt $STYLE-$STYLE \
        --lang-pairs src-tgt \
        --dae-styles $STYLE \
        --batch-size 128 \
        --skip-invalid-size-inputs-valid-test \
        --beam 5 \
        --lenpen 1.0 \
        --min-len 2 \
        --max-len-b 30 \
        --unkpen 3 \
        --no-repeat-ngram-size 3 \
        2>&1 | tee $MODEL_DIR/output_src_tgt.txt

grep ^S $MODEL_DIR/output_src_tgt.txt | cut -f2- | sed 's/ ##//g' > $MODEL_DIR/src.txt
grep ^T $MODEL_DIR/output_src_tgt.txt | cut -f2- | sed 's/ ##//g' > $MODEL_DIR/tgt.txt
grep ^H $MODEL_DIR/output_src_tgt.txt | cut -f3- | sed 's/ ##//g' > $MODEL_DIR/hypo_src_tgt.txt
cat $MODEL_DIR/hypo_src_tgt.txt | sacrebleu $MODEL_DIR/tgt.txt > $MODEL_DIR/log_sacrebleu_src_tgt
files2rouge $MODEL_DIR/hypo_src_tgt.txt $MODEL_DIR/tgt.txt > $MODEL_DIR/log_rouge_src_tgt

python files_merger.py $MODEL_DIR src.txt,hypo_src_tgt.txt,tgt.txt > $MODEL_DIR/output\_src\_F

cp $DATA_DIR/test.src-tgt.src.bin $DATA_DIR/test.src-$STYLE.src.bin
cp $DATA_DIR/test.src-tgt.src.idx $DATA_DIR/test.src-$STYLE.src.idx
cp $DATA_DIR/test.src-tgt.tgt.bin $DATA_DIR/test.src-$STYLE.$STYLE.bin
cp $DATA_DIR/test.src-tgt.tgt.idx $DATA_DIR/test.src-$STYLE.$STYLE.idx
cp $DATA_DIR/dict.src.txt $DATA_DIR/dict.$STYLE.txt

fairseq-generate $DATA_DIR \
        --path $MODEL \
        --user-dir mass \
        --task translation_mix \
        --model_lang_pairs src-tgt $STYLE-$STYLE \
        --lang-pairs src-$STYLE \
        --dae-styles $STYLE \
        --batch-size 128 \
        --skip-invalid-size-inputs-valid-test \
        --beam 5 \
        --lenpen 1.0 \
        --min-len 2 \
        --max-len-b 30 \
        --unkpen 3 \
        --no-repeat-ngram-size 3 \
        2>&1 | tee $MODEL_DIR/output_src\_$STYLE.txt

grep ^S $MODEL_DIR/output_src\_$STYLE.txt | cut -f2- | sed 's/ ##//g' > $MODEL_DIR/src.txt
grep ^T $MODEL_DIR/output_src\_$STYLE.txt | cut -f2- | sed 's/ ##//g' > $MODEL_DIR/tgt.txt
grep ^H $MODEL_DIR/output_src\_$STYLE.txt | cut -f3- | sed 's/ ##//g' > $MODEL_DIR/hypo_src_$STYLE.txt
cat $MODEL_DIR/hypo_src_$STYLE.txt | sacrebleu $MODEL_DIR/tgt.txt > $MODEL_DIR/log_sacrebleu\_src\_$STYLE
files2rouge $MODEL_DIR/hypo_src_$STYLE.txt $MODEL_DIR/tgt.txt > $MODEL_DIR/log_rouge\_src\_$STYLE

python files_merger.py $MODEL_DIR src.txt,hypo_src_$STYLE.txt,tgt.txt > $MODEL_DIR/output\_src\_$STYLE
