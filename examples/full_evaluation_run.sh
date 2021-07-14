python step_1_evauation_keras_models.py -path beijing_air_2_5 -end 2500 -model dnn
python step_1_evauation_keras_models.py -path beijing_air_2_5 -end 2500 -model cnn
python step_1_evauation_keras_models.py -path beijing_air_2_5 -end 2500 -model rnn

python step_2_perturbation_keras_models.py -path beijing_air_2_5 -model dnn
python step_2_perturbation_keras_models.py -path beijing_air_2_5 -model cnn
python step_2_perturbation_keras_models.py -path beijing_air_2_5 -model rnn

python step_3_perturbation_to_score.py -path beijing_air_2_5

python step_1_evauation_keras_models.py -path beijing_air_multi_site -end 2500 -model dnn
python step_1_evauation_keras_models.py -path beijing_air_multi_site -end 2500 -model cnn
python step_1_evauation_keras_models.py -path beijing_air_multi_site -end 2500 -model rnn

python step_2_perturbation_keras_models.py -path beijing_air_multi_site -model dnn
python step_2_perturbation_keras_models.py -path beijing_air_multi_site -model cnn
python step_2_perturbation_keras_models.py -path beijing_air_multi_site -model rnn

python step_3_perturbation_to_score.py -path beijing_air_multi_site

python step_1_evauation_keras_models.py -path metro_interstate -end 2500 -model dnn
python step_1_evauation_keras_models.py -path metro_interstate -end 2500 -model cnn
python step_1_evauation_keras_models.py -path metro_interstate -end 2500 -model rnn

python step_2_perturbation_keras_models.py -path metro_interstate -model dnn
python step_2_perturbation_keras_models.py -path metro_interstate -model cnn
python step_2_perturbation_keras_models.py -path metro_interstate -model rnn

python step_3_perturbation_to_score.py -path metro_interstate
