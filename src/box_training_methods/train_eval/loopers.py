from __future__ import annotations

import time
import os
import json
import math
from itertools import permutations
from pathlib import Path
from typing import *

import attr
import numpy as np
import networkx as nx
import torch
from loguru import logger
from scipy.sparse import coo_matrix
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm

# import torchmetrics
from box_training_methods.graph_modeling.metrics.mlc_metrics import MeanAvgPrecision, MicroAvgPrecision

from pytorch_utils.exceptions import StopLoopingException
from pytorch_utils.loggers import Logger
from pytorch_utils.training import IntervalConditional

from box_training_methods.metrics import *
from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz

# ### VISUALIZATION IMPORTS ONLY
# from box_training_methods.visualization.plot_2d_tbox import plot_2d_tbox
# from box_training_methods.models.box import TBox
# from box_training_methods.graph_modeling.dataset import edges_and_num_nodes_from_npz, create_positive_edges_from_tails, RandomNegativeEdges, HierarchicalNegativeEdges
# neg_sampler_obj_to_str = {
#     RandomNegativeEdges: "random",
#     HierarchicalNegativeEdges: "hierarchical"
# }
# ###


__all__ = [
    "GraphModelingTrainLooper",
    "MultilabelClassificationTrainLooper",
    "GraphModelingEvalLooper",
    "MultilabelClassificationEvalLooper"
]


@attr.s(auto_attribs=True)
class GraphModelingTrainLooper:
    name: str
    model: Module
    dl: DataLoader
    opt: torch.optim.Optimizer
    loss_func: Callable
    exact_negative_sampling: bool = False
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    eval_loopers: Iterable[EvalLooper] = attr.ib(factory=tuple)
    early_stopping: Callable = lambda z: None
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None
    save_model: Callable[Module] = lambda z: None
    log_interval: Optional[Union[IntervalConditional, int]] = attr.ib(
        default=None, converter=IntervalConditional.interval_conditional_converter
    )
    wordnet_save_model_tags: dict = None

    def __attrs_post_init__(self):
        if isinstance(self.eval_loopers, GraphModelingEvalLooper) or \
                isinstance(self.eval_loopers, MultilabelClassificationEvalLooper):
            self._eval_loopers = (self.eval_loopers,)
        self.looper_metrics = {"Total Examples": 0}
        if self.log_interval is None:
            # by default, log every batch
            self.log_interval = IntervalConditional(0)
        self.running_losses = []

        self.best_metrics_comparison_functions = {"Mean Loss": min}
        self.best_metrics = {}
        self.previous_best = None

    def loop(self, epochs: int):
        try:
            self.running_losses = []
            # box_collection = []
            for epoch in trange(epochs, desc=f"[{self.name}] Epochs"):
                self.model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)

                    ## SAVE WORDNET MODEL CHECKPOINT
                    if epoch % 10 == 0:
                        model_save_dir = "/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/icml2024_wordnet_models/"
                        filename = f'wordnet_full.epoch={epoch}-{self.wordnet_save_model_tags["model_type"]}-{self.wordnet_save_model_tags["negative_sampler"]}.pt'
                        model_save_fpath = model_save_dir + filename
                        logger.info(f"saving wordnet model to {model_save_fpath}")
                        model_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        torch.save(model_state_dict, model_save_fpath)
                        logger.info("saved wordnet model checkpoint")

                    # for eval_looper in self.eval_loopers:
                    #     eval_looper.loop(epoch=epoch, save_dir=self.save_model.run_dir)

                    # # 2D TBOX VISUALIZATION INFO
                    # if isinstance(self.model, TBox):
                    #     box_collection.append(torch.clone(self.model.boxes.detach()))

            # # VISUALIZE TBOX IN 2D
            # if isinstance(self.model, TBox):
            #     plot_2d_tbox(box_collection=torch.stack(box_collection),
            #                     negative_sampler=neg_sampler_obj_to_str[type(self.dl.dataset.negative_sampler)],
            #                     lr=self.opt.param_groups[0]['lr'],
            #                     negative_sampling_strategy=self.dl.dataset.negative_sampler.sampling_strategy if isinstance(self.dl.dataset.negative_sampler, HierarchicalNegativeEdges) else None)
        except StopLoopingException as e:
            logger.warning(str(e))
        finally:
            self.logger.commit()

            # load in the best model
            previous_device = next(iter(self.model.parameters())).device
            self.model.load_state_dict(self.save_model.best_model_state_dict())
            self.model.to(previous_device)

            # evaluate
            metrics = []
            predictions_coo = []
            for eval_looper in self.eval_loopers:
                metric, prediction_coo = eval_looper.loop(epoch='final', save_dir=self.save_model.run_dir)
                metrics.append(metric)
                predictions_coo.append(prediction_coo)
            return metrics, predictions_coo

    def train_loop(self, epoch: Optional[int] = None):
        """
        Internal loop for a single epoch of training
        :return: list of losses per batch
        """
        examples_this_epoch = 0
        examples_in_single_epoch = len(self.dl.dataset)
        last_time_stamp = time.time()
        num_batch_passed = 0
        for iteration, batch_in in enumerate(
            tqdm(self.dl, desc=f"[{self.name}] Batch", leave=False)
        ):
            self.opt.zero_grad()

            negative_padding_mask = None
            if self.exact_negative_sampling:
                batch_in, negative_padding_mask = torch.split(batch_in, (batch_in.shape[1] // 2) + 1, dim=1)
                negative_padding_mask = negative_padding_mask[..., 0].float()   # deduplicate

            batch_out = self.model(batch_in)
            if negative_padding_mask is not None:
                loss = self.loss_func(batch_out, negative_padding_mask=negative_padding_mask)
            else:
                loss = self.loss_func(batch_out)

            # This is not always going to be the right thing to check.
            # In a more general setting, we might want to consider wrapping the DataLoader in some way
            # with something which stores this information.
            num_in_batch = len(loss)

            loss = loss.sum(dim=0)

            self.looper_metrics["Total Examples"] += num_in_batch
            examples_this_epoch += num_in_batch

            if torch.isnan(loss).any():
                raise StopLoopingException("NaNs in loss")
            self.running_losses.append(loss.detach().item())
            loss.backward()

            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        raise StopLoopingException("NaNs in grad")

            num_batch_passed += 1
            # TODO: Refactor the following
            self.opt.step()
            # If you have a scheduler, keep track of the learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                if len(self.opt.param_groups) == 1:
                    self.looper_metrics[f"Learning Rate"] = self.opt.param_groups[0][
                        "lr"
                    ]
                else:
                    for i, param_group in enumerate(self.opt.param_groups):
                        self.looper_metrics[f"Learning Rate (Group {i})"] = param_group[
                            "lr"
                        ]

            # Check performance every self.log_interval number of examples
            last_log = self.log_interval.last

            if self.log_interval(self.looper_metrics["Total Examples"]):
                current_time_stamp = time.time()
                time_spend = (current_time_stamp - last_time_stamp) / num_batch_passed
                last_time_stamp = current_time_stamp
                num_batch_passed = 0
                self.logger.collect({"avg_time_per_batch": time_spend})

                self.logger.collect(self.looper_metrics)
                mean_loss = sum(self.running_losses) / (
                    self.looper_metrics["Total Examples"] - last_log
                )
                metrics = {"Mean Loss": mean_loss}
                self.logger.collect(
                    {
                        **{
                            f"[{self.name}] {metric_name}": value
                            for metric_name, value in metrics.items()
                        },
                        "Epoch": epoch + examples_this_epoch / examples_in_single_epoch,
                    }
                )
                self.logger.commit()
                self.running_losses = []
                self.update_best_metrics_(metrics)
                self.save_if_best_(self.best_metrics["Mean Loss"])
                self.early_stopping(self.best_metrics["Mean Loss"])

    def update_best_metrics_(self, metrics: Dict[str, float]) -> None:
        for name, comparison in self.best_metrics_comparison_functions.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = metrics[name]
            else:
                self.best_metrics[name] = comparison(
                    metrics[name], self.best_metrics[name]
                )
        self.summary_func(
            {
                f"[{self.name}] Best {name}": val
                for name, val in self.best_metrics.items()
            }
        )

    def save_if_best_(self, best_metric) -> None:
        if best_metric != self.previous_best:
            self.save_model(self.model)
            self.previous_best = best_metric


@attr.s(auto_attribs=True)
class MultilabelClassificationTrainLooper:
    name: str
    box_model: Module
    instance_model: Module
    dl: DataLoader
    opt: torch.optim.Optimizer
    loss_func: Callable
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    eval_loopers: Iterable[EvalLooper] = attr.ib(factory=tuple)
    early_stopping: Callable = lambda z: None
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None
    save_box_model: Callable[Module] = lambda z: None
    save_instance_model: Callable[Module] = lambda z: None
    log_interval: Optional[Union[IntervalConditional, int]] = attr.ib(
        default=None, converter=IntervalConditional.interval_conditional_converter
    )
    bioasq: bool = True

    def __attrs_post_init__(self):
        if isinstance(self.eval_loopers, GraphModelingEvalLooper) or \
                isinstance(self.eval_loopers, MultilabelClassificationEvalLooper):
            self._eval_loopers = (self.eval_loopers,)
        self.looper_metrics = {"Total Examples": 0}
        if self.log_interval is None:
            # by default, log every batch
            self.log_interval = IntervalConditional(0)
        self.running_losses = []

        self.best_metrics_comparison_functions = {"Mean Loss": min}
        self.best_metrics = {}
        self.previous_best = None

        if self.bioasq:
            self.train_loop = self.bioasq_train_loop
        else:
            self.train_loop = self.mlc_train_loop

    def loop(self, epochs: int):
        try:
            self.running_losses = []
            for epoch in trange(epochs, desc=f"[{self.name}] Epochs"):
                self.box_model.train()
                self.instance_model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)

                    # evaluate after each epoch
                    for eval_looper in self.eval_loopers:
                        if eval_looper.name == "Validation":
                            eval_looper.loop()

        except StopLoopingException as e:
            logger.warning(str(e))
        finally:
            self.logger.commit()

            # load in the best models
            previous_device = next(iter(self.box_model.parameters())).device
            self.box_model.load_state_dict(self.save_box_model.best_model_state_dict())
            self.box_model.to(previous_device)

            self.instance_model.load_state_dict(self.save_instance_model.best_model_state_dict())
            self.instance_model.to(previous_device)

            # TODO!!!
            # evaluate
            metrics = []
            predictions_coo = []
            # for eval_looper in self.eval_loopers:
            #     metric, prediction_coo = eval_looper.loop()
            #     metrics.append(metric)
            #     predictions_coo.append(prediction_coo)
            return metrics, predictions_coo

    def biaosq_train_loop(self, epoch: Optional[int] = None):
        """
        Internal loop for a single epoch of training
        :return: list of losses per batch
        """

        last_time_stamp = time.time()
        num_batch_passed = 0

        logger.info(f'Start looping epoch {epoch}')

        batch_times = []
        batch_sum = 0
        step = 0

        begin_dl_iter = time.time()
        dl_iter = iter(self.dl)
        
        while True:

            try:
                begin_batch = time.time()
                self.opt.zero_grad()

                begin_next = time.time()
                batch = next(dl_iter)
                end_next = time.time()
                logger.critical(f"\nnext took {end_next - begin_next} seconds")

                inputs, positives, negatives = batch
                # logger.warning(f"inputs: {inputs['input_ids'].shape}")
                # logger.warning(f"positives: {positives.shape}")
                # logger.warning(f"negatives: {negatives.shape}")

                begin_nn = time.time()
                input_encs = self.instance_model(inputs)
                
                positive_boxes = self.box_model.boxes[positives]  # (batch_size, num_positives, 2, dim)
                negative_boxes = self.box_model.boxes[negatives]  # (batch_size, num_positives, 2, dim)
                positive_energy = self.box_model.scores(instance_boxes=input_encs, 
                                                        label_boxes=positive_boxes, 
                                                        intersection_temp=0.01, 
                                                        volume_temp=1.0)
                negative_energy = self.box_model.scores(instance_boxes=input_encs,
                                                        label_boxes=negative_boxes, 
                                                        intersection_temp=0.01, 
                                                        volume_temp=1.0)
                end_nn = time.time()
                logger.critical(f"NNs took {end_nn - begin_nn} seconds")

                # TODO input-label scorer
                begin_loss_back = time.time()
                loss = self.loss_func(log_prob_pos=positive_energy, log_prob_neg=negative_energy)
                loss = loss.sum(dim=0)

                if torch.isnan(loss).any():
                    raise StopLoopingException("NaNs in loss")
                self.running_losses.append(loss.detach().item())

                loss.backward()
                end_loss_back = time.time()
                logger.critical(f"Loss and backward took {end_loss_back - begin_loss_back} seconds")

                num_batch_passed += 1
                self.opt.step()

                end_batch = time.time()

                batch_delta = end_batch - begin_batch
                batch_times.append(batch_delta)
                step += 1
                batch_sum += batch_delta
                logger.critical(f"Batch took {batch_delta} seconds")        
                logger.critical(f"Batch running avg: {batch_sum / step}")

                if step % 1000 == 0:
                    self.save_models(epoch=epoch, step=step)

            except StopIteration:
                break

        end_dl_iter = time.time()
        logger.critical(f"Iterating through dl took {str(end_dl_iter - begin_dl_iter)} seconds")

    def mlc_train_loop(self, epoch: Optional[int] = None):
        """
        Internal loop for a single epoch of training
        :return: list of losses per batch
        """

        last_time_stamp = time.time()
        num_batch_passed = 0
        losses_for_epoch = []

        logger.info(f'Start looping epoch {epoch}')

        for batch in self.dl:

            self.opt.zero_grad()

            feats, positives, positives_pad_mask, negatives, negatives_pad_mask = batch
            feat_encs = self.instance_model(feats)

            positive_boxes = self.box_model.boxes[positives]  # (batch_size, num_positives, 2, dim)
            negative_boxes = self.box_model.boxes[negatives]  # (batch_size, num_positives, 2, dim)
            positive_energy = self.box_model.scores(instance_boxes=feat_encs, 
                                                    label_boxes=positive_boxes, 
                                                    intersection_temp=0.01, 
                                                    volume_temp=1.0)
            negative_energy = self.box_model.scores(instance_boxes=feat_encs,
                                                    label_boxes=negative_boxes, 
                                                    intersection_temp=0.01, 
                                                    volume_temp=1.0)

            # TODO input-label scorer
            loss = self.loss_func(log_prob_pos=positive_energy, log_prob_neg=negative_energy, positive_padding_mask=positives_pad_mask, negative_padding_mask=negatives_pad_mask)
            loss = loss.sum(dim=0)

            if torch.isnan(loss).any():
                raise StopLoopingException("NaNs in loss")
            self.running_losses.append(loss.detach().item())

            losses_for_epoch.append(loss.detach().item())
            loss.backward()

            num_batch_passed += 1
            self.opt.step()

        logger.critical(f"Average loss for epoch {epoch}: {sum(losses_for_epoch)/len(losses_for_epoch)}")

    def update_best_metrics_(self, metrics: Dict[str, float]) -> None:
        for name, comparison in self.best_metrics_comparison_functions.items():
            if name not in self.best_metrics:
                self.best_metrics[name] = metrics[name]
            else:
                self.best_metrics[name] = comparison(
                    metrics[name], self.best_metrics[name]
                )
        self.summary_func(
            {
                f"[{self.name}] Best {name}": val
                for name, val in self.best_metrics.items()
            }
        )

    def save_if_best_(self, best_metric) -> None:
        if best_metric != self.previous_best:
            self.save_box_model(self.box_model)
            self.save_instance_model(self.instance_model)
            self.previous_best = best_metric

    def save_models(self, epoch, step) -> None:
        
        logger.critical(f"box_model run_dir: {self.save_box_model.run_dir}")
        self.save_box_model(self.box_model)
        self.save_box_model.run_dir = Path("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/bioasq_models")
        self.save_box_model.filename = f'tbox.epoch-{epoch}.step-{step}.pt'
        self.save_box_model.save_to_disk(None)
        
        logger.critical(f"instance_model run_dir: {self.save_instance_model.run_dir}")
        self.save_instance_model(self.instance_model)
        self.save_instance_model.run_dir = Path("/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/bioasq_models")
        self.save_instance_model.filename = f'embeddings.epoch-{epoch}.step-{step}.pt'
        self.save_instance_model.save_to_disk(None)


def batched_pairs(num_nodes, batch_size):
  batch = []
  for row_idx, col_idx in permutations(range(num_nodes), 2):
    batch.append([row_idx, col_idx])
    if len(batch) == batch_size:
      yield torch.tensor(batch, dtype=torch.long)
      batch = []
  if batch:
    yield torch.tensor(batch, dtype=torch.long)
    

@attr.s(auto_attribs=True)
class GraphModelingEvalLooper:
    name: str
    model: Module
    dl: DataLoader
    batchsize: int
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None
    output_dir: str = None
    no_f1_save_matrices: bool = False       # flag to be enabled for WordNet which runs out of memory at F1 step — save matrices to disk
    model_checkpoint_fpath: str = None      # pass along model checkpoint for WordNet eval to parse its name

    @torch.no_grad()
    def loop(self, epoch: Optional[int] = None, save_dir: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()

        ## WORDNET EVAL SAVE MATRICES BEFORE CALCULATING F1, CALCULATE F1 SEPARATELY ON CPU WITH MORE MEMORY
        if self.no_f1_save_matrices:
            logger.debug("a")
            previous_device = next(iter(self.model.parameters())).device
            logger.debug("b")
            ground_truth = np.zeros((82115, 82115))
            logger.debug("c")
            pos_index, _ = edges_and_num_nodes_from_npz("/project/pi_mccallum_umass_edu/brozonoyer_umass_edu/graph-data/realworld/wordnet_full/wordnet_full.npz")
            logger.debug("d")
            ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1
            logger.debug("e")
            prediction_scores = np.zeros((82115, 82115))
            logger.debug("f")
            with torch.no_grad():
                for batch_idxs in tqdm(batched_pairs(82115, self.batchsize), desc=f"Evaluating", total = math.ceil(82115 * (82115 - 1) / self.batchsize)):
                    cur_preds = self.model(batch_idxs.to(previous_device)).cpu().numpy()
                    prediction_scores[batch_idxs[:,0], batch_idxs[:,1]] = cur_preds
            prediction_scores_no_diag = prediction_scores[~np.eye(82115, dtype=bool)]
            logger.debug("g")
            ground_truth_no_diag = ground_truth[~np.eye(82115, dtype=bool)]
            logger.debug("h")
            ckpt_info = self.model_checkpoint_fpath.split("/")[-1].split(".")[-2]       # e.g. "epoch=20-vector_sim-hierarchical"
            logger.debug("i")
            save_preds_fpath = f"/scratch/workspace/wenlongzhao_umass_edu-hans/icml2024_wordnet_prediction_scores_no_diag/{ckpt_info}.npy"
            logger.debug("j")
            np.save(save_preds_fpath, prediction_scores_no_diag.astype(np.float16))
            logger.debug("k")
            return
        ######################################################################################################

        logger.debug("Evaluating model predictions on full adjacency matrix")
        time1 = time.time()
        previous_device = next(iter(self.model.parameters())).device
        # num_nodes = self.dl.dataset.num_nodes
        num_nodes = self.dl.sampler.data_source.num_nodes
        ground_truth = np.zeros((num_nodes, num_nodes))
        # pos_index = self.dl.dataset.edges.cpu().numpy()
        if self.dl.sampler.data_source.graph_npz_file is not None:
            # for the HANS graph modeling experiments, this will load original TC graph regardless of whether training_edges are TC or TR
            pos_index, _ = edges_and_num_nodes_from_npz(self.dl.sampler.data_source.graph_npz_file)
        else:
            pos_index = self.dl.sampler.data_source.edges_tc.cpu().numpy()
        # # release RAM
        # del self.dl.dataset

        ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1
        prediction_scores = np.zeros((num_nodes, num_nodes))
        with torch.no_grad():
            for batch_idxs in tqdm(batched_pairs(num_nodes, self.batchsize), desc=f"Evaluating", total = math.ceil(num_nodes * (num_nodes - 1) / self.batchsize)):
                cur_preds = self.model(batch_idxs.to(previous_device)).cpu().numpy()
                prediction_scores[batch_idxs[:,0], batch_idxs[:,1]] = cur_preds

        logger.debug("removing diag from prediction scores")
        prediction_scores_no_diag = prediction_scores[~np.eye(num_nodes, dtype=bool)]
        logger.debug("removing diag from ground truth")
        ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]
        
        time2 = time.time()
        logger.debug(f"Evaluation time: {time2 - time1}")

        logger.debug("Calculating optimal F1 score")
        metrics = calculate_optimal_F1(ground_truth_no_diag, prediction_scores_no_diag)
        time3 = time.time()
        logger.debug(f"F1 calculation time: {time3 - time2}")
        logger.info(f"Metrics: {metrics}")

        self.logger.collect({f"[{self.name}] {k}": v for k, v in metrics.items()})
        self.logger.commit()

        predictions = (prediction_scores > metrics["threshold"]) * (
            ~np.eye(num_nodes, dtype=bool)
        )

        # if save_dir is not None:
            
        #     predictions_path = os.path.join(save_dir, f'predictions.epoch-{epoch}.npy')
        #     with open(predictions_path, 'wb') as f:
        #         np.save(f, coo_matrix(predictions), allow_pickle=True)
        #     logger.info(f"Saving predictions to: {predictions_path}")

        #     prediction_scores_path = os.path.join(save_dir, f'prediction_scores.epoch-{epoch}.npy')
        #     with open(prediction_scores_path, 'wb') as f:
        #         np.save(f, prediction_scores, allow_pickle=True)
        #     logger.info(f"Saving prediction_scores to: {prediction_scores_path}")
            
        #     metrics_path = os.path.join(save_dir, f'metrics.epoch-{epoch}.json')
        #     with open(metrics_path, 'w') as f:
        #         json.dump(metrics, f, indent=4, sort_keys=True)
        #     logger.info(f"Saving metrics to: {metrics_path}")
        
        return metrics, coo_matrix(predictions)


@attr.s(auto_attribs=True)
class MultilabelClassificationEvalLooper:
    name: str
    box_model: Module
    instance_model: Module
    dl: DataLoader
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None

    @torch.no_grad()
    def loop(self) -> Dict[str, Any]:
        self.instance_model.eval()
        self.box_model.eval()

        dl_iter = iter(self.dl)
        # mlap = torchmetrics.classification.MultilabelAveragePrecision(num_labels=len(self.dl.dataset.label_encoder.classes_), average='micro')
        micro_map = MicroAvgPrecision()
        instance_map = MeanAvgPrecision()

        while True:

            try:
                batch = next(dl_iter)
                inputs, labels = batch
                input_encs = self.instance_model(inputs)                                         # (batch_size, 2, dim)
                label_boxes = self.box_model.boxes.unsqueeze(dim=0)                              # (1, num_labels, 2, dim)
                label_boxes = label_boxes.repeat(input_encs.shape[0], 1, 1, 1)                   # (batch_size, num_labels, 2, dim)
                energy = self.box_model.scores(instance_boxes=input_encs,
                                               label_boxes=label_boxes,
                                               intersection_temp=0.01,
                                               volume_temp=1.0)
                # TODO compute predictions from energy score

                # metrics = calculate_optimal_F1(torch.flatten(labels).numpy(), torch.flatten(energy).cpu().numpy())
                instance_map(-energy, labels)
                micro_map(-energy, labels)

            except StopIteration:
                break

        instance_map_value = instance_map.get_metric(reset=True)
        micro_map_value = micro_map.get_metric(reset=True)
        logger.critical(f"instance_map_value: {instance_map_value}")
        logger.critical(f"micro_map_value: {micro_map_value}")
