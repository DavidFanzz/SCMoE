Unchosen Experts Can Contribute Too: Unleashing MoE Models' Power by Self-Contrast
===
## Setup
### Requirements
```
pip install datasets
pip install evaluate
pip install absl-py
pip install nltk
pip install pylint
pip install antlr4-python3-runtime==4.11.1
pip install transformers==4.40.0
```
### Add modeling models


Replace the configuration_mixtral.py, modeling_mixtral.py in the transformers/src/transfomers/models/mixtral with .py file in ./modeling_models/Mixtral
Replace the config.json file in model's config file with config.json in ./modeling_models/Mixtral

### GPU Requirements

For Mixtral, you should need 4 A100 40G or 2 A100 80G. 
For DeepSeekMoE, you should need 2 A100 40G or 1 A100 80G.

## Inference
`--decoding_method` refers to a certain method in (greedy, dynamic, cs, dola, cd, scmoe).
`--num_experts_per_tok` refers to number of activation experts for MoE Model or the strong activation of SCMoE. Default to be 2 for Mixtral\
`--student_num_experts_per_tok` refers to number of activation experts for the weak activation of SCMoE.
`--routed_tok` refers to the routed expert rank id for weak activation. id are begin from 0 to 7 for Mixtral when using rank-$k$ routing
`--cd_beta` refers to the parameter $\beta$ in SCMoE



```
task=gsm8k
model_name=Mixtral
decoding_method=scmoe
python3 generate.py\
    --decoding_method ${decoding_method}\
    --infile ./data/${task}/input.jsonl\
    --outfile ./results/${task}/${model_name}_${decoding_method}.jsonl\
    --model ${model_name}\
    --cd_beta ${cd3_beta}\
    --gpus_per_model 4\
    --world_size 4\
    --batch_size 1\
    --num_experts_per_tok 2\
    --student_num_experts_per_tok 1\
    --student_routed_tok 2\
    --max_new_tokens 512 
```
## Evaluation
task=gsm8k
model_name=Mixtral
decoding_method=scmoe
```
python3 evaluation.py\
    --model ${model_name}\
    --task_name ${task}\
    --load_generations_path ./results/${task}/${model_name}_${decoding_method}.jsonl\
    --metric_output_path ./results/${task}_results.jsonl\
    --allow_code_execution
```