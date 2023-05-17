import click
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



@click.command(context_settings=dict(show_default=True),)
@click.option("--model_type", type=click.Choice(['tbox', 'vector_sim']), required=True, help="model type")
@click.option("--graph_type", type=click.Choice([
    'balanced_tree/branching=10-log_num_nodes=13',
    'balanced_tree/branching=2-log_num_nodes=13',
    'balanced_tree/branching=3-log_num_nodes=13',
    'balanced_tree/branching=5-log_num_nodes=13',
    'nested_chinese_restaurant_process/alpha=10-log_num_nodes=13',
    'nested_chinese_restaurant_process/alpha=100-log_num_nodes=13',
    'nested_chinese_restaurant_process/alpha=500-log_num_nodes=13',
    'price/c=0.01-gamma=1.0-log_num_nodes=13-m=1',
    'price/c=0.01-gamma=1.0-log_num_nodes=13-m=10',
    'price/c=0.01-gamma=1.0-log_num_nodes=13-m=5',
    'price/c=0.1-gamma=1.0-log_num_nodes=13-m=1',
    'price/c=0.1-gamma=1.0-log_num_nodes=13-m=10',
    'price/c=0.1-gamma=1.0-log_num_nodes=13-m=5'
]), help="graph type", required=True)
@click.option("--tc_or_tr", type=click.Choice(['tc', 'tr']), required=True, help="transitive closure or transitive reduction")
@click.option("--negative_sampler", type=click.Choice(['hierarchical', 'random']), required=True, help="sampling method")
@click.option("--negative_ratio", type=int, required=True, help="negative ratio")
@click.option(
    "--seed", type=int, help="seed for random number generator mostly for the model",
)
@click.option("--graph_seed", type=int, help="seed for random number generator mostly for the graph")
def train_final(**config):
    """A new entry point for the final runs before neurips"""
    from .train import training
    final_config = {
        'task': 'graph_modeling',
        'negative_permutation_option': 'none',
        'dim': 64,
        'log_batch_size': 9,
        'log_eval_batch_size': 17,
        'epochs': 40,
        'margin': 1.0,
        'hierarchical_negative_sampling_strategy': 'uniform',
        'patience': 1000,
        'log_interval': 0.2,
        'eval': True,
        'cuda': True,
        'save_predictions': False,
        'wandb': True,
        'vector_separate_io': True,
        'vector_use_bias': True,
        'box_intersection_temp': 0.01,
        'box_volume_temp': 1.0,
        'tbox_temperature_type': 'global',
        'save_model': False,
        'constrain_deltas_fn': 'sqr',
        'undirected': None,
    }
    graphs_dir = '/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/data/graphs13/'
    if config['model_type'] == 'tbox':
        final_config['model_type'] = 'tbox'
        final_config['learning_rate'] = 0.2
        final_config['negative_weight'] = 0.9
    elif config['model_type'] == 'vector_sim':
        import json
        with open("./best_runs.jsonl", "r") as f:
            best_runs = [json.loads(l.strip()) for l in f.readlines()]

        final_config['model_type'] = 'vector_sim'
        best_run = [r for r in best_runs if config['graph_type'] in r['data_path']]

        try:
            assert len(best_run) == 1
        except AssertionError:
            best_run = best_run[0]
            print("Multiple best runs found, using the first one")
            print(best_run)

        final_config['learning_rate'] = best_run['learning_rate']
        final_config['negative_weight'] = best_run['negative_weight']
    if config['tc_or_tr'] == 'tc':
        config["graph_type"] = '-'.join([config["graph_type"], "transitive_closure=True"])
    elif config['tc_or_tr'] == 'tr':
        config["graph_type"] = '-'.join([config["graph_type"], "transitive_closure=False"])
    else:
        raise ValueError("tc_or_tr must be one of 'tc' or 'tr'")
    if 'balanced_tree' in config['graph_type']:
        if 'branching=10' in config['graph_type'] and 'transitive_closure=True' in config['graph_type']:
            seed_map = {
                1: 415728013,
            }

        elif 'branching=10' in config['graph_type'] and 'transitive_closure=False' in config['graph_type']:
            seed_map = {
                1: 2150935259
            }
        elif 'branching=2' in config['graph_type'] and 'transitive_closure=True' in config['graph_type']:
            seed_map = {
                1: 1901635484,
            }
        elif 'branching=2' in config['graph_type'] and 'transitive_closure=False' in config['graph_type']:
            seed_map = {
                1: 2902554954,
            }
        elif 'branching=3' in config['graph_type'] and 'transitive_closure=True' in config['graph_type']:
            seed_map = {
                1: 1439248948,
            }
        elif 'branching=3' in config['graph_type'] and 'transitive_closure=False' in config['graph_type']:
            seed_map = {
                1: 38311313
            }
        elif 'branching=5' in config['graph_type'] and 'transitive_closure=True' in config['graph_type']:
            seed_map = {
                1: 1246911898
            }
        elif 'branching=5' in config['graph_type'] and 'transitive_closure=False' in config['graph_type']:
            seed_map = {
                1: 367229542
            }
        else:
            raise ValueError("Balanced tree only has branching 2, 3, 5, 10")



        if config["graph_seed"] not in seed_map:
            raise ValueError("Balanced tree only has seeds 1")
        config['graph_seed'] = seed_map[config['graph_seed']]

    final_config['data_path'] = str((Path(graphs_dir) / config['graph_type'] / str(config['graph_seed'])).with_suffix('.npz'))

    final_config['negative_sampler'] = config['negative_sampler']
    final_config['negative_ratio'] = config['negative_ratio']
    final_config['seed'] = config['seed']
    final_config['sample_positive_edges_from_tc_or_tr'] = config['tc_or_tr']

    for k, v in config.items():
        final_config[f"g_{k}"] = v
    # wandb
    if final_config['wandb']:
        final_config['wandb_tags'] = [
            f"model_type={final_config['model_type']}" , 
            f"negative_sampler={final_config['negative_sampler']}",
            f"negative_ratio={final_config['negative_ratio']}",
            f"seed={final_config['seed']}",
            f"tc_or_tr={config['tc_or_tr']}",
            f"graph_seed={config['graph_seed']}"
        ]
        gt, gp = config['graph_type'].split('/')
        final_config['wandb_tags'].append(f"gt")
        for k_v in gp.split('-'):
            final_config['wandb_tags'].append(k_v)
        final_config['wandb_name'] = f"final_run-{final_config['model_type']}"
    final_config['output_dir'] = None
    training(final_config)
    
