from __future__ import annotations

import time
import os
import json
from typing import *

import attr
import numpy as np
import torch
from loguru import logger
from scipy.sparse import coo_matrix
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange, tqdm
from itertools import chain

from pytorch_utils.exceptions import StopLoopingException
from pytorch_utils.loggers import Logger
from pytorch_utils.training import IntervalConditional

from box_training_methods.metrics import *

### VISUALIZATION IMPORTS ONLY
from box_training_methods.visualization.plot_2d_tbox import plot_2d_tbox
from box_training_methods.models.box import TBox
from box_training_methods.graph_modeling.dataset import create_positive_edges_from_tails, RandomNegativeEdges, HierarchicalNegativeEdges
neg_sampler_obj_to_str = {
    RandomNegativeEdges: "random",
    HierarchicalNegativeEdges: "hierarchical"
}
###


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
            box_collection = []
            for epoch in trange(epochs, desc=f"[{self.name}] Epochs"):
                self.model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)

                    # save model at epoch to wandb.run.dir (e.g. /work/pi_mccallum_umass_edu/brozonoyer_umass_edu/box-training-methods/wandb/run-20230403_234846-86c2gllp/files/)
                    self.save_model.filename = f'learned_model.epoch-{epoch}.pt'
                    self.save_model.save_to_disk(None)
                    for eval_looper in self.eval_loopers:
                        eval_looper.loop(epoch=epoch, save_dir=self.save_model.run_dir)

                    # 2D TBOX VISUALIZATION INFO
                    if isinstance(self.model, TBox):
                        box_collection.append(torch.clone(self.model.boxes.detach()))

            # VISUALIZE TBOX IN 2D
            if isinstance(self.model, TBox):
                plot_2d_tbox(box_collection=torch.stack(box_collection),
                                negative_sampler=neg_sampler_obj_to_str[type(self.dl.dataset.negative_sampler)],
                                lr=self.opt.param_groups[0]['lr'],
                                negative_sampling_strategy=self.dl.dataset.negative_sampler.sampling_strategy if isinstance(self.dl.dataset.negative_sampler, HierarchicalNegativeEdges) else None)
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
            loss = self.loss_func(batch_out, negative_padding_mask=negative_padding_mask)

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
    scorer: Module
    dl: DataLoader
    opt: torch.optim.Optimizer
    label_label_loss_func: Callable
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
            for epoch in trange(epochs, desc=f"[{self.name}] Epochs"):
                self.box_model.train()
                self.instance_model.train()
                with torch.enable_grad():
                    self.train_loop(epoch)
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

    def train_loop(self, epoch: Optional[int] = None):
        """
        Internal loop for a single epoch of training
        :return: list of losses per batch
        """

        last_time_stamp = time.time()
        num_batch_passed = 0

        logger.info(f'Start looping epoch {epoch}')

        for batch in self.dl:

            inputs, positives, negatives = batch

            input_encs = self.instance_model(inputs)
            breakpoint()

            # # TODO RandomNegativeEdges currently doesn't store adjacency matrix
            # positive_label_label_idxs = create_positive_edges_from_tails(tails=label_batch_in.cpu(), A=self.label_label_dl.dataset.negative_sampler.A)  # FIXME only HierarchicalNegativeEdges has A attribute, not RandomNegativeEdges
            # negative_label_label_idxs = self.label_label_dl.dataset.negative_sampler(positive_label_label_idxs)
            # label_label_batch_in = torch.cat([positive_label_label_idxs.unsqueeze(1), negative_label_label_idxs], dim=1)

            # self.opt.zero_grad()

            # num_in_batch = instance_batch_in.shape[0]
            # self.looper_metrics["Total Examples"] += num_in_batch
            # examples_this_epoch += num_in_batch

            # # compute L_G for labels related to instance
            # label_label_batch_out = self.box_model(label_label_batch_in)
            # label_label_loss = self.label_label_loss_func(label_label_batch_out).sum(dim=0)

            # # compute instance encoding
            # # TODO need to tailor loss function to instance-label without negative samples
            # # TODO pass in padding mask for exact hierarchical negative sampling
            # instance_boxes = self.instance_model(instance_batch_in)     # (bsz, 2 [-/+], dim)
            # instance_label_batch_out = self.box_model.forward(idxs=label_batch_in, instances=instance_boxes)
            
            # # FIXME currently the loss fn expects a specified shape with negatives
            # # instance_label_loss = self.label_label_loss_func(instance_label_batch_out).sum(dim=0)

            # loss = label_label_loss # + instance_label_loss

            # if torch.isnan(loss).any():
            #     raise StopLoopingException("NaNs in loss")
            # self.running_losses.append(loss.detach().item())
            # loss.backward()

            # for param in chain(self.box_model.parameters(), self.instance_model.parameters()):
            #     if param.grad is not None:
            #         if torch.isnan(param.grad).any():
            #             raise StopLoopingException("NaNs in grad")

            # num_batch_passed += 1
            # # TODO: Refactor the following
            # self.opt.step()
            # # If you have a scheduler, keep track of the learning rate
            # if self.scheduler is not None:
            #     self.scheduler.step()
            #     if len(self.opt.param_groups) == 1:
            #         self.looper_metrics[f"Learning Rate"] = self.opt.param_groups[0][
            #             "lr"
            #         ]
            #     else:
            #         for i, param_group in enumerate(self.opt.param_groups):
            #             self.looper_metrics[f"Learning Rate (Group {i})"] = param_group[
            #                 "lr"
            #             ]

            # # Check performance every self.log_interval number of examples
            # last_log = self.log_interval.last

            # if self.log_interval(self.looper_metrics["Total Examples"]):
            #     current_time_stamp = time.time()
            #     time_spend = (current_time_stamp - last_time_stamp) / num_batch_passed
            #     last_time_stamp = current_time_stamp
            #     num_batch_passed = 0
            #     self.logger.collect({"avg_time_per_batch": time_spend})

            #     self.logger.collect(self.looper_metrics)
            #     mean_loss = sum(self.running_losses) / (
            #         self.looper_metrics["Total Examples"] - last_log
            #     )
            #     metrics = {"Mean Loss": mean_loss}
            #     self.logger.collect(
            #         {
            #             **{
            #                 f"[{self.name}] {metric_name}": value
            #                 for metric_name, value in metrics.items()
            #             },
            #             "Epoch": epoch + examples_this_epoch / examples_in_single_epoch,
            #         }
            #     )
            #     self.logger.commit()
            #     self.running_losses = []
            #     self.update_best_metrics_(metrics)
            #     self.save_if_best_(self.best_metrics["Mean Loss"])
            #     self.early_stopping(self.best_metrics["Mean Loss"])

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


@attr.s(auto_attribs=True)
class GraphModelingEvalLooper:
    name: str
    model: Module
    dl: DataLoader
    batchsize: int
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None
    output_dir: str = None

    @torch.no_grad()
    def loop(self, epoch: Optional[int] = None, save_dir: Optional[str] = None) -> Dict[str, Any]:
        self.model.eval()

        logger.debug("Evaluating model predictions on full adjacency matrix")
        time1 = time.time()
        previous_device = next(iter(self.model.parameters())).device
        # num_nodes = self.dl.dataset.num_nodes
        num_nodes = self.dl.sampler.data_source.num_nodes
        ground_truth = np.zeros((num_nodes, num_nodes))
        # pos_index = self.dl.dataset.edges.cpu().numpy()
        pos_index = self.dl.sampler.data_source.edges.cpu().numpy()
        # # release RAM
        # del self.dl.dataset

        ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1

        prediction_scores = np.zeros((num_nodes, num_nodes))  # .to(previous_device)

        input_x, input_y = np.indices((num_nodes, num_nodes))
        input_x, input_y = input_x.flatten(), input_y.flatten()
        input_list = np.stack([input_x, input_y], axis=-1)
        number_of_entries = len(input_x)

        with torch.no_grad():
            pbar = tqdm(
                desc=f"[{self.name}] Evaluating", leave=False, total=number_of_entries
            )
            cur_pos = 0
            while cur_pos < number_of_entries:
                last_pos = cur_pos
                cur_pos += self.batchsize
                if cur_pos > number_of_entries:
                    cur_pos = number_of_entries

                ids = torch.tensor(input_list[last_pos:cur_pos], dtype=torch.long)
                cur_preds = self.model(ids.to(previous_device)).cpu().numpy()
                prediction_scores[
                    input_x[last_pos:cur_pos], input_y[last_pos:cur_pos]
                ] = cur_preds
                pbar.update(self.batchsize)

        prediction_scores_no_diag = prediction_scores[~np.eye(num_nodes, dtype=bool)]
        ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]

        time2 = time.time()
        logger.debug(f"Evaluation time: {time2 - time1}")

        # TODO: release self.dl from gpu
        del input_x, input_y

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

        if save_dir is not None:
            
            predictions_path = os.path.join(save_dir, f'predictions.epoch-{epoch}.npy')
            with open(predictions_path, 'wb') as f:
                np.save(f, coo_matrix(predictions), allow_pickle=True)
            logger.info(f"Saving predictions to: {predictions_path}")

            prediction_scores_path = os.path.join(save_dir, f'prediction_scores.epoch-{epoch}.npy')
            with open(prediction_scores_path, 'wb') as f:
                np.save(f, prediction_scores, allow_pickle=True)
            logger.info(f"Saving prediction_scores to: {prediction_scores_path}")
            
            metrics_path = os.path.join(save_dir, f'metrics.epoch-{epoch}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4, sort_keys=True)
            logger.info(f"Saving metrics to: {metrics_path}")
        
        return metrics, coo_matrix(predictions)


@attr.s(auto_attribs=True)
class MultilabelClassificationEvalLooper:
    name: str
    model: Module
    dl: DataLoader
    batchsize: int
    logger: Logger = attr.ib(factory=Logger)
    summary_func: Callable[Dict] = lambda z: None

    @torch.no_grad()
    def loop(self) -> Dict[str, Any]:
        self.model.eval()

        logger.debug("Evaluating model predictions on full adjacency matrix")
        time1 = time.time()
        previous_device = next(iter(self.model.parameters())).device
        num_nodes = self.dl.dataset.num_nodes
        ground_truth = np.zeros((num_nodes, num_nodes))
        pos_index = self.dl.dataset.edges.cpu().numpy()
        # release RAM
        del self.dl.dataset

        ground_truth[pos_index[:, 0], pos_index[:, 1]] = 1

        prediction_scores = np.zeros((num_nodes, num_nodes))  # .to(previous_device)

        input_x, input_y = np.indices((num_nodes, num_nodes))
        input_x, input_y = input_x.flatten(), input_y.flatten()
        input_list = np.stack([input_x, input_y], axis=-1)
        number_of_entries = len(input_x)

        with torch.no_grad():
            pbar = tqdm(
                desc=f"[{self.name}] Evaluating", leave=False, total=number_of_entries
            )
            cur_pos = 0
            while cur_pos < number_of_entries:
                last_pos = cur_pos
                cur_pos += self.batchsize
                if cur_pos > number_of_entries:
                    cur_pos = number_of_entries

                ids = torch.tensor(input_list[last_pos:cur_pos], dtype=torch.long)
                cur_preds = self.model(ids.to(previous_device)).cpu().numpy()
                prediction_scores[
                    input_x[last_pos:cur_pos], input_y[last_pos:cur_pos]
                ] = cur_preds
                pbar.update(self.batchsize)

        prediction_scores_no_diag = prediction_scores[~np.eye(num_nodes, dtype=bool)]
        ground_truth_no_diag = ground_truth[~np.eye(num_nodes, dtype=bool)]

        time2 = time.time()
        logger.debug(f"Evaluation time: {time2 - time1}")

        # TODO: release self.dl from gpu
        del input_x, input_y

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

        return metrics, coo_matrix(predictions)
