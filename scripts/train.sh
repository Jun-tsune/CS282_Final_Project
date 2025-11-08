#!/bin/bash
export PYTHONPATH=$(pwd)
python src/models/transformer_backbone.py # test the transformer implementation
python src/models/full_models.py # test the model implementation