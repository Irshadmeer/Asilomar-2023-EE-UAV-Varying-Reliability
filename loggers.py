from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class SummaryWriterCallback(BaseCallback):
    """
    Snippet skeleton from Stable baselines3 documentation here:
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#directly-accessing-the-summary-writer
    """

    def _on_training_start(self):
        self._log_freq = 10  # log every 10 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        self.tb_formatter = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )

    def _on_step(self) -> bool:
        """
        Log my_custom_reward every _log_freq(th) to tensorboard for each environment
        """
        if self.n_calls % self._log_freq == 0:
            _info_dict = self.locals["infos"][0]
            _rewards = [
                "total_power",
                "power_fraction",
                "reward",
                "outage_violations",
                "reward_ee",
                "reward_reliability",
                "num_active_BS",
            ]
            for _reward_name in _rewards:
                self.tb_formatter.writer.add_scalar(
                    f"rewards/{_reward_name}", _info_dict[_reward_name], self.n_calls
                )
