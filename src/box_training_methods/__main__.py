import click

# graph_modeling is the only task that needs to generate graphs, so treat graph_modeling.generate as top-level group
from .graph_modeling.generate.__main__ import main as generate
from .train_eval.__main__ import train, eval, train_final, train_1


@click.group()
def main():
    """Scripts to generate graphs, train and evaluate graph representations"""
    pass


main.add_command(generate, "generate")
main.add_command(train, "train")
main.add_command(eval, "eval")
main.add_command(train_final, "train_final")

# entrypoints for ICML 2024 experiments (4 different types of sweeps for model x negative sampling combinations)
main.add_command(vector_sim_hyperparameter_tuning, "vector_sim_hyperparameter_tuning")  # vector_sim, random negative sampling sweep to determine best lr and nw per graph type
main.add_command(train_tbox, "train_tbox")  # tbox, both random and hierarchical negative sampling
main.add_command(train_vector_sim, "train_vector_sim")  # vector_sim, both random and hierarchical negative sampling, using best lr and nw values from train_vector_sim_hyperparameter_tuning sweep

if __name__ == "__main__":
    main()
