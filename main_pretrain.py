import os
import json
import random
from pprint import pprint

from pytorch_lightning import (Trainer, seed_everything)
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint,
        EarlyStopping)
from pytorch_lightning.loggers import (WandbLogger, CSVLogger)
from pytorch_lightning.strategies.ddp import DDPStrategy

from ula.args.setup import parse_args_pretrain
from ula.data.classification_dataloader import prepare_val_data
from ula.data.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_n_crop_transform,
)
from ula.data.augmentations import prepare_train_transform
from ula.methods import METHODS
from ula.utils.misc import (make_contiguous, EMACallback)


try:
    from orion.client import cli as orion_cli
except ImportError:
    _orion_available = False
else:
    _orion_available = True


def main():
    args = parse_args_pretrain()
    if args.seed == -1:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)
    seed_everything(args.seed, workers=True)

    transform_kwargs = (
        args.transform_kwargs if args.unique_augs > 1 else [args.transform_kwargs]
    )

    # validation dataloader for when it is available
    if args.valid_data_path is None:
        val_loader = None
    else:
        val_loader = []
        if args.dataset in ['celeba', 'waterbirds']:
            val_iters = [(args.valid_data_path[0], 'valid'),
                         (args.valid_data_path[0], 'test')]
        else:
            val_iters = [(path, 'valid') for path in args.valid_data_path]

        for i, (valid_data_path, split) in enumerate(val_iters):
            val_loader_, _ = prepare_val_data(
                args.dataset,
                split=split,
                valid_data_path=valid_data_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                **transform_kwargs[0]  # get crop_size, mean, std
            )
            val_loader.append(val_loader_)

    transform = prepare_n_crop_transform(
        [prepare_train_transform(args.dataset, augment=args.augment, **kwargs) for kwargs in transform_kwargs],
        num_crops_per_aug=args.num_crops_per_aug,
    )

    if args.debug_augmentations:
        print("Transforms:")
        pprint(transform)

    train_dataset = prepare_datasets(
        args.dataset,
        transform,
        train_data_path=args.train_data_path,
    )
    train_loader = prepare_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Build model
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"

    if args.num_large_crops != 2:
        assert args.method in ["wmse", "mae"]

    print(args.__dict__)
    model = METHODS[args.method](train_dataset=train_dataset, **args.__dict__)
    make_contiguous(model)

    # Auto-resuming
    ckpt_path = None
    if args.auto_resume and args.resume_from_checkpoint is None:
        last_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
        if os.path.exists(last_ckpt_dir):
            ckpt_path = last_ckpt_dir
    elif args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        del args.resume_from_checkpoint

    callbacks = []

    # Logging
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.wandb:
        import wandb

        job_type = 'pretrain' if not args.test_only else 'online_test'
        logger = WandbLogger(
            name=args.name,
            save_dir=str(args.checkpoint_dir),
            offline=args.offline,
            resume="allow",
            id=args.name + '_' + job_type,
            job_type=job_type
        )
        logger.watch(model, log="gradients", log_freq=100)
        logger.log_hyperparams(args)

        # lr logging
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
    else:
        logger = CSVLogger(save_dir=args.checkpoint_dir, name='pretrain')
        logger.log_hyperparams(args)

    if args.save_checkpoint:
        json_path = os.path.join(args.checkpoint_dir, "args.json")
        with open(json_path, 'w') as f:
            json.dump(vars(args), f, default=lambda o: "<not serializable>")

        select_best_model = args.model_selection_metric is not None and args.select_best_model
        model_ckpt_cb = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='checkpoint_{epoch}',
            save_last=True, save_top_k=-1,
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
            trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
        except RuntimeError:
            if _orion_available:
                orion_cli.report_bad_trial()
            raise
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
                    **transform_kwargs[0]  # get crop_size, mean, std
                )
                test_loader.append(test_loader_)

            if select_best_model:
                best_ckpt_dir = model_ckpt_cb.best_model_path
            else:
                best_ckpt_dir = os.path.join(args.checkpoint_dir, 'last.ckpt')
            trainer.test(model, ckpt_path=str(best_ckpt_dir), dataloaders=test_loader)


if __name__ == "__main__":
    main()
