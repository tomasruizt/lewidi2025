import json_repair
import pandas as pd
from typing import Iterable
from pydantic_settings import BaseSettings
import seaborn as sns


class Args(BaseSettings, cli_parse_args=True):
    file: str


def main():
    args = Args()
    train_df, eval_df = parse_dfs(args.file)
    dump_plot_loss(train_df, eval_df)


def parse_dfs(file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = list(load_rows(file))
    train_rows = [row for row in all_rows if "loss" in row]
    eval_rows = [row for row in all_rows if "eval_loss" in row]
    train_df = pd.DataFrame(train_rows)
    eval_df = pd.DataFrame(eval_rows)
    return train_df, eval_df


def load_rows(file: str) -> Iterable[dict]:
    with open(file, "rt") as f:
        for line in f:
            if line.startswith("{'"):
                yield json_repair.loads(line)


def dump_plot_loss(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> None:
    joint_df = pd.concat(
        [
            train_df[["loss", "epoch"]].assign(series="train"),
            eval_df[["eval_loss", "epoch"]]
            .assign(series="eval")
            .rename(columns={"eval_loss": "loss"}),
        ]
    )

    ax = sns.lineplot(
        data=joint_df.iloc[1:],
        x="epoch",
        y="loss",
        hue="series",
        marker="o",
    )
    ax.grid(alpha=0.5)
    ax.figure.savefig("loss.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
