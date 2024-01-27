import click

# graph_modeling is the only task that needs to generate graphs, so treat graph_modeling.generate as top-level group
from .graph_modeling.generate.__main__ import main as generate
from .train_eval.__main__ import train, eval, hyperparameter_tuning, synthetic_graphs, wordnet_full, wordnet_full_eval


@click.group()
def main():
    """Scripts to generate graphs, train and evaluate graph representations"""
    pass


main.add_command(generate, "generate")
main.add_command(train, "train")
main.add_command(eval, "eval")

# entrypoints for ICML 2024 experiments
main.add_command(hyperparameter_tuning, "hyperparameter_tuning")  # determine best lr and nw
main.add_command(synthetic_graphs, "synthetic_graphs")
main.add_command(wordnet_full, "wordnet_full")
main.add_command(wordnet_full_eval, "wordnet_full_eval")

if __name__ == "__main__":
    main()
