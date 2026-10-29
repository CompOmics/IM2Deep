"""Architecture callbacks."""

import logging

import lightning as L

try:
    import wandb  # type: ignore[import] # ty: ignore[unresolved-import]
except ImportError:
    wandb = None

LOGGER = logging.getLogger(__name__)

#: Attributes of a transfer model that hold the pretrained feature branches.
#: The concatenation head is deliberately excluded: it is what fine-tuning is
#: meant to adapt.
BACKBONE_BRANCHES = ("ConvAtomComp", "ConvDiatomComp", "ConvGlobal", "OneHot", "MolDesc")


class LogLowestMAE(L.Callback):
    def __init__(self, config):
        super().__init__()
        self.bestMAE = float("inf")
        self.config = config

    def on_validation_end(self, trainer, pl_module):
        try:
            currentMAE = trainer.callback_metrics["Validation MAE"]
        except KeyError:  # Multi
            currentMAE = trainer.callback_metrics["Val Mean MAE"]
        if currentMAE < self.bestMAE:
            self.bestMAE = currentMAE
        if self.config["wandb"]["enabled"]:
            if wandb is not None:
                wandb.log({"Best Val MAE": self.bestMAE})


class BackboneFreeze(L.Callback):
    """
    Freeze a transfer model's pretrained feature branches for a warmup.

    Mirrors DeepLC's ``freeze_epochs``/``unfreeze_lr_scale`` fine-tuning
    behaviour: the pretrained branches are held fixed while the concatenation
    head adapts, then everything is unfrozen and training continues at a
    reduced learning rate so the warmed-up head is not immediately undone.

    Parameters
    ----------
    freeze_epochs
        Number of epochs to keep the feature branches frozen. ``0`` disables
        the callback entirely.
    unfreeze_lr_scale
        Factor applied to every optimizer learning rate on unfreezing.

    """

    def __init__(self, freeze_epochs: int = 0, unfreeze_lr_scale: float = 0.1):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.unfreeze_lr_scale = unfreeze_lr_scale

    def _set_requires_grad(self, pl_module: L.LightningModule, requires_grad: bool) -> None:
        for name in BACKBONE_BRANCHES:
            branch = getattr(pl_module, name, None)
            if branch is None:
                continue
            for parameter in branch.parameters():
                parameter.requires_grad = requires_grad

    def on_train_start(self, trainer, pl_module) -> None:
        if self.freeze_epochs > 0:
            self._set_requires_grad(pl_module, False)
            LOGGER.info(
                f"Froze pretrained feature branches for the first {self.freeze_epochs} epoch(s)."
            )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if self.freeze_epochs <= 0 or trainer.current_epoch != self.freeze_epochs:
            return
        self._set_requires_grad(pl_module, True)
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] *= self.unfreeze_lr_scale
        LOGGER.info(
            "Unfroze pretrained feature branches and scaled the learning rate by "
            f"{self.unfreeze_lr_scale}."
        )
