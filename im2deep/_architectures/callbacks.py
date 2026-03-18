"""Architecture callbacks."""

import lightning as L

try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None


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
