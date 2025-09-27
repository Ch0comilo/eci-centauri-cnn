This is a variation of the Liang model found in https://github.com/yuliang419/Astronet-Vetting.

#### Basic commands

Generate new input data:
```
python -m astronet.preprocess.generate_input_records --input_tce_csv_file="C:\Users\danie\data\NASA\tces_with_labels_v3.csv" --tess_data_dir="C:\Users\danie\data\NASA\LC" --output_dir=astronet/tfrecords-new+old --num_shards=8
```

Train ensemble (modify the .sh file to match the output_dir above):
```
./astronet/ensemble_train.sh
```

Tune (requires some setup, see Tune.ipynb):
```
python astronet/tune.py --model=AstroCNNModel --config_name=local_global_new --train_files=astronet/tfrecords-new\+old/test-0000[0-5]* --eval_files=astronet/tfrecords-new\+old/test-0000[6-6]* --train_steps=7000 --tune_trials=1000 --client_secrets=${HOME}/client_secrets.json --study_id=a_unique_string_id
```

Run predictions (or use Predict.ipynb for one-offs):
```
python astronet/predict.py --model_dir=/tmp/astronet/AstroCNNModel_local_global_multiclass_20200222_154634 --data_files=astronet/tfrecords-new\+old/* --output_file=/home/${USER}/predictions.csv
```
