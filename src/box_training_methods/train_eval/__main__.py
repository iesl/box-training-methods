import click
import json
import copy
from pathlib import Path


class IntOrPercent(click.ParamType):
    name = "click_union"

    def convert(self, value, param, ctx):
        try:
            float_value = float(value)
            if 0 <= float_value <= 1:
                return float_value
            elif float_value == int(float_value):
                return int(float_value)
            else:
                self.fail(
                    f"expected float between [0,1] or int, got {float_value}",
                    param,
                    ctx,
                )
        except TypeError:
            self.fail(
                "expected string for int() or float() conversion, got "
                f"{value!r} of type {type(value).__name__}",
                param,
                ctx,
            )
        except ValueError:
            self.fail(f"{value!r} is not a valid integer or float", param, ctx)


# TODO weed out the task-specific arguments from here and move them to task-specific train_eval methods

@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--task",
    type=click.Choice(["graph_modeling", "multilabel_classification", "bioasq"], case_sensitive=False),
    help="task to train on",
    required=True
)
@click.option(
    "--data_path",
    type=click.Path(),
    help="directory or file with data (eg. data/graph/some_tree)",
    required=True,
)
@click.option(
    "--mesh_parent_child_mapping_path",
    type=click.Path(),
    help="parent-child mapping text file, e.g. 'MeSH_parent_child_mapping_2020.txt'",
)
@click.option(
    "--mesh_name_id_mapping_path",
    type=click.Path(),
    help="MeSH name-id mapping text file, e.g. 'MeSH_name_id_mapping_2020.txt'",
)
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "tbox",
            "gumbel_box",
            "hard_box",
            "order_embeddings",
            "partial_order_embeddings",
            "vector_sim",
            "vector_dist",
            "bilinear_vector",
            "complex_vector",
            "lorentzian_distance",
            "lorentzian_score",
            "lorentzian",
            "hyperbolic_entailment_cones",
        ],
        case_sensitive=False,
    ),
    default="tbox",
    help="model architecture to use",
)
@click.option(
    "--negatives_permutation_option",
    type=click.Choice(["none", "head", "tail"], case_sensitive=False),
    default="none",
    help="whether to use permuted negatives during training, and if so whether to permute head or tail",
)
@click.option(
    "--undirected / --directed",
    default=None,
    help="whether to train using an undirected or directed graph (default is model dependent)",
    show_default=False,
)
@click.option(
    "--dim", type=int, default=4, help="dimension for embedding space",
)
@click.option(
    "--log_batch_size",
    type=int,
    default=10,
    help="batch size for training will be 2**LOG_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--log_eval_batch_size",
    type=int,
    default=15,
    help="batch size for eval will be 2**LOG_EVAL_BATCH_SIZE",
)  # Using batch sizes which are 2**n for some integer n may help optimize GPU efficiency
@click.option(
    "--learning_rate", type=float, default=0.01, help="learning rate",
)
@click.option(
    "--negative_weight", type=float, default=0.9, help="weight of negative loss",
)
@click.option(
    "--margin",
    type=float,
    default=1.0,
    help="margin for MaxMarginWithLogitsNegativeSamplingLoss or BCEWithDistancesNegativeSamplingLoss (unused otherwise)",
)
@click.option(
    "--negative_sampler",
    type=str,
    default="random",
    help="whether to use RandomNegativeEdges or HierarchicalNegativeEdges"
)
@click.option(
    "--hierarchical_negative_sampling_strategy",
    type=str,
    default="exact",
    help="which negative edges to sample and with what probability to sample them"
)
@click.option(
    "--negative_ratio",
    type=int,
    default=128,
    help="number of negative samples for each positive",
)
@click.option(
    "--epochs", type=int, default=1_000, help="maximum number of epochs to train"
)
@click.option(
    "--patience",
    type=int,
    default=11,
    help="number of log_intervals without decreased loss before stopping training",
)
@click.option(
    "--log_interval",
    type=IntOrPercent(),
    default=0.1,
    help="interval or percentage (as float in [0,1]) of examples to train between logging training metrics",
)
@click.option(
    "--eval / --no_eval",
    default=True,
    help="whether or not to evaluate the model at the end of training",
)
@click.option(
    "--cuda / --no_cuda", default=True, help="enable/disable CUDA (eg. no nVidia GPU)",
)
@click.option(
    "--save_prediction / --no_save_prediction",
    default=False,
    help="enable/disable saving predicted adjacency matrix",
)
@click.option(
    "--seed", type=int, help="seed for random number generator",
)
@click.option(
    "--wandb / --no_wandb",
    default=False,
    help="enable/disable logging to Weights and Biases",
)
@click.option(
    "--vector_separate_io / --vector_no_separate_io",
    default=True,
    help="enable/disable using separate input/output representations for vector / bilinear vector model",
)
@click.option(
    "--vector_use_bias / --vector_no_use_bias",
    default=False,
    help="enable/disable using bias term in vector / bilinear",
)
@click.option(
    "--lorentzian_alpha",
    type=float,
    default=5.0,
    help="penalty for distance, where higher alpha emphasises distance as a determining factor in edge direction more",
)
@click.option(
    "--lorentzian_beta",
    type=float,
    default=1.0,
    help="-1/curvature of the space, if beta is higher the space is less curved / more euclidean",
)
@click.option(
    "--hyperbolic_entailment_cones_relative_cone_aperture_scale",
    type=float,
    default=1.0,
    help="float in (0,1) representing relative scale of cone apertures with respect to radius (K = relative_cone_aperature_scale * eps_bound / (1 - eps_bound^2))",
)
@click.option(
    "--hyperbolic_entailment_cones_eps_bound",
    type=float,
    default=0.1,
    help="restrict vectors to be parameterized in an annulus from eps to 1-eps",
)
@click.option(
    "--constrain_deltas_fn",
    type=click.Choice(["sqr", "exp", "softplus", "proj"]),
    default="sqr",
    help="which function to apply to width parameters of hard_box in order to make them positive, or use projected gradient descent (clipping in forward method)"
)
@click.option(
    "--box_intersection_temp",
    type=float,
    default=0.01,
    help="temperature of intersection calculation (hyperparameter for gumbel_box, initialized value for tbox)",
)
@click.option(
    "--box_volume_temp",
    type=float,
    default=1.0,
    help="temperature of volume calculation (hyperparameter for gumbel_box, initialized value for tbox)",
)
@click.option(
    "--tbox_temperature_type",
    type=click.Choice(["global", "per_dim", "per_entity", "per_entity_per_dim"]),
    default="per_entity_per_dim",
    help="type of learned temperatures (for tbox model)",
)
@click.option(
    "--output_dir",
    type=str,
    default=None,
    help="output directory for recording current hyper-parameters and results",
)
@click.option(
    "--save_model / --no_save_model",
    type=bool,
    default=False,
    help="whether or not to save the model to disk",
)
def train(**config):
    """Train an embedding representation on a task with boxes"""
    from .train import training

    training(config)


@click.command(context_settings=dict(show_default=True),)
def eval():
    """Evaluate an embedding representation on a task with boxes"""
    pass


def parse_graph_path(path):  # e.g. /project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/graphs13/price/c=0.01-gamma=1.0-log_num_nodes=13-m=5-transitive_closure=True(/4.npz)
    path = path.strip()
    pieces = path.rstrip("/").split("/")
    if path.endswith(".npz"):
        graph_type, graph_hparams, graph_seed = pieces[-3], pieces[-2], pieces[-1][:-len(".npz")]
        graph_tags = {"graph_type": graph_type, "graph_seed": graph_seed}
    else:
        graph_type, graph_hparams = pieces[-2], pieces[-1]
        graph_tags = {"graph_type": graph_type}
    graph_hparams = [h.split("=") for h in graph_hparams.split("-")]
    graph_hparams = {h[0]: h[1] for h in graph_hparams}
    graph_tags.update(graph_hparams)
    return graph_tags


BASE_CONFIG = {
    'box_intersection_temp': 0.01,
    'box_volume_temp': 1.0,
    'cuda': True,
    'dim': 64,
    'epochs': 50,   # 12 to catch best-converging runs via best_run()
    'eval': True,
    'log_batch_size': 9,
    'log_eval_batch_size': 17,
    'log_interval': 0.2,
    'negatives_permutation_option': 'none',
    'output_dir': None,
    'patience': 1000,
    'save_model': False,
    'save_prediction': False,
    'seed': None,
    'task': 'graph_modeling',
    'tbox_temperature_type': 'global',
    'undirected': None,
    'vector_separate_io': True,
    'vector_use_bias': True,
    'wandb': True,
}


@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--data_path",
    type=click.Path(),
    help="directory or file with data",
)
@click.option(
    "--learning_rate", type=float, default=0.01, help="learning rate",
)
@click.option(
    "--model_type",
    type=click.Choice(["tbox", "vector_sim"])
)
@click.option(
    "--negative_ratio",
    type=int,
    default=128,
    help="number of negative samples for each positive",
)
@click.option(
    "--negative_sampler",
    type=click.Choice(["random", "hierarchical"])
)
@click.option(
    "--negative_weight", type=float, default=0.9, help="weight of negative loss",
)
@click.option(
    "--sample_positive_edges_from_tc_or_tr",
    type=click.Choice(['tc', 'tr']),
    required=True,
    help="sample positive edges from transitive closure or transitive reduction"
)
@click.option("--mesh", default=0)
def hyperparameter_tuning(**config):
    from .train import training
    final_config = copy.deepcopy(BASE_CONFIG)
    final_config.update(config)   
    if final_config["mesh"] != 1:
        graph_tags = parse_graph_path(config['data_path'])
        wandb_tags = ["=".join([k, str(v)]) for k, v in config.items() if k != 'data_path']
        wandb_tags.extend(["=".join([k, str(v)]) for k, v in graph_tags.items()])
        final_config['wandb_tags'] = wandb_tags
    training(final_config)


@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--data_path",
    type=click.Path(),
    help="directory or file with data",
)
@click.option(
    "--model_type",
    type=click.Choice(["tbox", "vector_sim"])
)
@click.option(
    "--negative_ratio",
    type=int,
    default=128,
    help="number of negative samples for each positive",
)
@click.option(
    "--negative_sampler",
    type=str,
    default="random",
    help="whether to use RandomNegativeEdges or HierarchyAwareNegativeEdges"
)
@click.option(
    "--sample_positive_edges_from_tc_or_tr",
    type=click.Choice(['tc', 'tr']),
    required=True,
    help="sample positive edges from transitive closure or transitive reduction"
)
@click.option(
    "--lr_nw_json", type=str, help="path to json storing graph type and negative ratio to best learning rate and negative weight"
)
def synthetic_graphs(**config):
    from .train import training
    final_config = copy.deepcopy(BASE_CONFIG)
    final_config["epochs"] = 40

    # --- get best learning_rate and negative_weight from json
    with open(config["lr_nw_json"], "r") as f:
        lr_nw_json = json.load(f)
    del config["lr_nw_json"]

    graph_type, graph_hparams = config["data_path"].split("/")[-3:-1]
    model_type = config["model_type"]
    negative_sampler = config["negative_sampler"]
    negative_ratio = config["negative_ratio"]
    sample_positive_edges_from_tc_or_tr = config["sample_positive_edges_from_tc_or_tr"]
    
    learning_rate = lr_nw_json[graph_type][graph_hparams][model_type][f'negative_sampler={negative_sampler}'][f'negative_ratio={negative_ratio}'][f'sample_positive_edges_from_{sample_positive_edges_from_tc_or_tr}']['best_learning_rate']
    negative_weight = lr_nw_json[graph_type][graph_hparams][model_type][f'negative_sampler={negative_sampler}'][f'negative_ratio={negative_ratio}'][f'sample_positive_edges_from_{sample_positive_edges_from_tc_or_tr}']['best_negative_weight']
    config.update({
        "learning_rate": learning_rate,
        "negative_weight": negative_weight,
    })
    # ------

    final_config.update(config)
    graph_tags = parse_graph_path(config['data_path'])
    wandb_tags = ["=".join([k, str(v)]) for k, v in config.items() if k != 'data_path']
    wandb_tags.extend(["=".join([k, str(v)]) for k, v in graph_tags.items()])
    final_config['wandb_tags'] = wandb_tags
    training(final_config)


@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--model_type",
    type=click.Choice(["tbox", "vector_sim"])
)
@click.option(
    "--negative_sampler",
    type=str,
    default="random",
    help="whether to use RandomNegativeEdges or HierarchyAwareNegativeEdges"
)
def wordnet_full(**config):
    from .train import training
    wordnet_config = {
        'box_intersection_temp': 0.01,
        'box_volume_temp': 1.0,
        'cuda': True,
        'data_path': '/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz',
        'dim': 64,
        'epochs': 500,
        'eval': True,
        'learning_rate': 0.1,
        'log_batch_size': 5,
        'log_eval_batch_size': 17,
        'log_interval': 0.2,
        'negative_ratio': 128,
        'negative_weight': 0.9,
        'negatives_permutation_option': 'none',
        'output_dir': None,
        'patience': 10000,
        'save_model': False,
        'save_prediction': False,
        'seed': None,
        'task': 'graph_modeling',
        'tbox_temperature_type': 'global',
        'undirected': None,
        'vector_separate_io': True,
        'vector_use_bias': True,
        'wandb': True,
    }
    wordnet_config.update(config)
    if config["negative_sampler"] == "hierarchical":
        wordnet_config["sample_positive_edges_from_tc_or_tr"] = "tr"
    else:
        wordnet_config["sample_positive_edges_from_tc_or_tr"] = "tc"
    training(wordnet_config)


@click.command(context_settings=dict(show_default=True),)
@click.option(
    "--model_type",
    type=click.Choice(["tbox", "vector_sim"])
)
@click.option(
    "--model_checkpoint",
    type=str,
)
def wordnet_full_eval(**config):
    from .eval import evaluation

    wordnet_eval_config = {        
        'box_intersection_temp': 0.01,
        'box_volume_temp': 1.0,
        'cuda': True,
        'data_path': '/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz',
        'dim': 64,
        'log_eval_batch_size': 17,
        "task": "graph_modeling",
        'tbox_temperature_type': 'global',
        'undirected': None,
        'vector_separate_io': True,
        'vector_use_bias': True,
        'wandb': False,
    }
    config.update(wordnet_eval_config)
    evaluation(config)
