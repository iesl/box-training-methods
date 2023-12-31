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
main.add_command(train_vector_sim_random, "train_vector_sim_random")  # vector_sim, random negative sampling
main.add_command(train_tbox, "train_tbox")  # tbox, both random and hierarchical negative sampling

if __name__ == "__main__":
    main()
