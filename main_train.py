import os
import random
import json
from pprint import pprint
import traceback

import torch
from torch import nn
from pytorch_lightning import (Trainer, seed_everything)
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
        EarlyStopping)
from pytorch_lightning.loggers import (WandbLogger, CSVLogger)
from pytorch_lightning.strategies.ddp import DDPStrategy

from ula.args.setup import parse_args_train
from ula.data.classification_dataloader import (
        prepare_train_data, prepare_val_data)
from ula.methods import SUPERVISED_METHODS
from ula.utils.misc import (make_contiguous, EMACallback)

try:
    from orion.client import cli as orion_cli
except ImportError:
    _orion_available = False
else:
    _orion_available = True


def main():
    args = parse_args_train()
    if args.seed == -1:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)
    seed_everything(args.seed, workers=True)

    class_loss_func = nn.CrossEntropyLoss()

    val_loader = []
    val_datasets = []
    if args.dataset in ['celeba', 'waterbirds']:
        val_iters = [(args.valid_data_path[0], 'valid'),
                        (args.valid_data_path[0], 'test')]
    else:
        val_iters = [(path, 'valid') for path in args.valid_data_path]

    for i, (valid_data_path, split) in enumerate(val_iters):
        val_loader_, val_dataset = prepare_val_data(
            args.dataset,
            split=split,
            valid_data_path=valid_data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **args.transform_kwargs
        )
        val_loader.append(val_loader_)
        val_datasets.append(val_dataset)

    train_loader, train_dataset = prepare_train_data(
        args.dataset,
        train_data_path=args.train_data_path,
        split=args.train_data_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        **args.transform_kwargs
        )

    # Create supervised model
    model = SUPERVISED_METHODS[args.method](
        train_dataset=train_dataset,
        valid_datasets=val_datasets,
        class_loss_func=class_loss_func,
        **vars(args))
    make_contiguous(model)

    callbacks = []

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_dir = str(args.checkpoint_dir)
    if args.wandb:
        import wandb

        job_type = 'train'
        logger = WandbLogger(
            name=args.name,
            save_dir=checkpoint_dir,
            offline=args.offline,
            resume="allow",
            id=args.name + '_' + job_type,
            job_type=job_type
        )
        #  logger.watch(model, log="gradients", log_freq=100)
        logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        logger = CSVLogger(save_dir=args.checkpoint_dir, name='train')
        logger.log_hyperparams(args)

    # Auto-resuming
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        last_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_dir):
            ckpt_path = last_ckpt_dir
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    if args.save_checkpoint:
        json_path = os.path.join(args.checkpoint_dir, "args.json")
        with open(json_path, 'w') as f:
            json.dump(vars(args), f, default=lambda o: "<not serializable>")

        select_best_model = args.model_selection_metric is not None and args.select_best_model
        model_ckpt_cb = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='checkpoint_{epoch}',
            save_last=True,
            save_top_k=1,
            monitor=args.model_selection_metric if select_best_model else None,
            mode=args.model_selection_mode,
            auto_insert_metric_name=False,
            save_weights_only=False,
            every_n_epochs=args.checkpoint_frequency,
            save_on_train_epoch_end=False
        )
        callbacks.append(model_ckpt_cb)

    # Early stopping callback
    if args.use_early_stopping:
        assert args.model_selection_metric is not None
        early_stopping_cb = EarlyStopping(
            args.model_selection_metric,
            mode=args.model_selection_mode,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            check_on_train_epoch_end=False
        )
        callbacks.append(early_stopping_cb)

    # EMA model selection metric for hyperparameter optimization
    if args.model_selection_metric is not None:
        hyperopt_ema_cb = EMACallback(
            args.model_selection_metric,
            momentum=args.hyperopt_metric_momentum
            )
        callbacks.append(hyperopt_ema_cb)

    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=True,
        strategy=DDPStrategy(find_unused_parameters=False)
        if args.strategy == "ddp"
        else args.strategy,
    )

    has_succeeded = False
    if not args.test_only:
        try:
            trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=ckpt_path)
        except RuntimeError as e:
            if _orion_available:
                orion_cli.report_bad_trial()
            print(traceback.format_exc())
        else:
            has_succeeded = True

            if _orion_available and args.model_selection_metric is not None:
                if select_best_model:
                    obj = float(model_ckpt_cb.best_model_score)
                elif args.use_early_stopping:
                    obj = float(early_stopping_cb.best_score)
                else:
                    obj = hyperopt_ema_cb.value

                # Orion minimize the following objective
                print('Final validation objective:', obj)
                if args.wandb:
                    wandb.run.summary[f"{args.model_selection_metric}/best"] = obj
                if args.model_selection_mode == 'max':
                    obj = 100 - obj
                orion_cli.report_objective(obj)

    if args.test_only or has_succeeded:
        if args.test_data_path is not None and args.save_checkpoint:
            test_loader = list()
            for test_data_path in args.test_data_path:

                test_loader_, _ = prepare_val_data(
                    args.dataset,
                    split='test',
                    valid_data_path=test_data_path,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    **args.transform_kwargs
                )
                test_loader.append(test_loader_)

            if select_best_model:
                best_ckpt_dir = model_ckpt_cb.best_model_path
            else:
                best_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
            trainer.test(model, ckpt_path=str(best_ckpt_dir), dataloaders=test_loader)


if __name__ == "__main__":
    main()
