#!/bin/bash
export PYTHONPATH=$(pwd)
# python src/models/transformer_backbone.py # test the transformer implementation
# python src/models/full_models.py # test the model implementation
python src/train.py model_yaml='config_model_1' train_yaml='config_train_1'


# #!/bin/bash
# # For Windows
# # initiate Conda（Windows Git Bash)
# source /d/Anaconda/etc/profile.d/conda.sh

# # activate Conda env
# conda activate env_282PJ

# export PYTHONPATH=$(pwd)

# # python src/models/transformer_backbone.py
# # python src/models/full_models.py
# python src/train.py model_yaml='config_model_2' train_yaml='config_train_1'