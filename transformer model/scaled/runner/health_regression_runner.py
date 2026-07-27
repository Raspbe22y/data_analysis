import random
import torch
from tqdm import tqdm
import time
import os
import torch.nn.functional as F
import numpy as np
from .runner import Runner


class HealthRegressionRunner(Runner):
    def __init__(self, config, model, dataset_info):
        super().__init__(config, model, dataset_info)

    def prepare_input(self, batch):
        x = batch[0].to(self.weight_dtype)  # (B, 300, 11)
        y = batch[1].to(self.weight_dtype)  # (B, 8)
        return x, y

    def setup_model(self):
        self.model.requires_grad_(True)

    def setup_optimizer(self):
        self.trainable_params = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )
        self.logger.info(f"Total trainable params {len(self.trainable_params)}")
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.learning_rate,
            betas=(self.cfg.solver.adam_beta1, self.cfg.solver.adam_beta2),
            weight_decay=self.cfg.solver.adam_weight_decay,
            eps=self.cfg.solver.adam_epsilon,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

    def log_validation(self, model):
        model.eval()
        with torch.no_grad():
            index = random.randint(0, len(self.val_dataset) - 1)
            x, y = self.val_dataset[index]
            x = x.to(self.weight_dtype).to(self.accelerator.device).unsqueeze(0)
            y = y.to(self.weight_dtype).to(self.accelerator.device).unsqueeze(0)
            pred = model(x)

        pred_np = pred.detach().cpu().numpy().squeeze(0)
        gt_np = y.detach().cpu().numpy().squeeze(0)
        self.logger.info(f"Validation sample {index}:")
        self.logger.info(f"  Prediction:   {np.round(pred_np, 4)}")
        self.logger.info(f"  Ground truth: {np.round(gt_np, 4)}")
        self.logger.info(f"  MAE:          {np.mean(np.abs(pred_np - gt_np)):.6f}")
        model.train()

    def run(self):
        progress_bar = tqdm(
            range(self.cfg.solver.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        progress_bar.update(self.global_step)

        for epoch in range(self.first_epoch, self.num_train_epochs):
            train_loss = 0.0
            t_data_start = time.time()

            for step, batch in enumerate(self.train_dataloader):
                t_data = time.time() - t_data_start

                with self.accelerator.accumulate(self.model):
                    x, y = self.prepare_input(batch)
                    pred = self.model(x)
                    loss = F.mse_loss(pred.float(), y.float(), reduction="mean")

                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.cfg.train_bs)
                    ).mean()
                    train_loss += (
                        avg_loss.item() / self.cfg.solver.gradient_accumulation_steps
                    )

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_params, self.cfg.solver.max_grad_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1
                    self.accelerator.log(
                        {"train_loss": train_loss}, step=self.global_step
                    )
                    train_loss = 0.0

                    if (self.global_step % self.cfg.val.validation_steps == 0) or (
                        self.global_step in self.cfg.val.validation_steps_tuple
                    ):
                        if self.accelerator.is_main_process:
                            unwrap_net = self.accelerator.unwrap_model(self.model)
                            self.save_checkpoint(
                                unwrap_net, self.save_dir, "model",
                                self.global_step, total_limit=4,
                            )
                            self.log_validation(unwrap_net)

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.learning_rate,
                    "td": f"{t_data:.2f}s",
                }
                t_data_start = time.time()
                progress_bar.set_postfix(**logs)

                if self.global_step >= self.cfg.solver.max_train_steps:
                    break

            self.accelerator.wait_for_everyone()
