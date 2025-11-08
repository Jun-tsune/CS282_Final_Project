# CS282-Fall2025 Final Project

This is the repository for the final project of CS282-Fall2025.

This project studies how compressive memory affects In-Context Learning (ICL) in longsequence Transformers. Using synthetic tasks, we compare standard and Compressive Transformers across context lengths and memory ratios. We expect compressive memory to enhance long-range ICL while showing diminishing gains as compression increases.

The project is based on the following repositories:
* [What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](https://github.com/dtsip/in-context-learning)
* [Compressive Transformers for Long-Range Sequence Modelling](https://github.com/lucidrains/compressive-transformer-pytorch)


## File Structure
* `src/`: Source code for models and training.
* `notebooks/`: Jupyter notebooks for experiments and analysis.
* `output/`: Directory to save model checkpoints and results.
* `data/`: Directory for datasets (if applicable).
* `scripts/`: Bash scripts for running experiments.

