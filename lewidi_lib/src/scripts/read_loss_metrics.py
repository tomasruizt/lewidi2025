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
    train_loss_df = (
        train_df[["loss", "epoch"]]
        .assign(series="train", variable="loss")
        .rename(columns={"loss": "value"})
    )
    eval_loss_df = (
        eval_df[["eval_loss", "epoch"]]
        .assign(series="eval", variable="loss")
        .rename(columns={"eval_loss": "value"})
    )
    grad_norm_df = (
        train_df[["grad_norm", "epoch"]]
        .assign(series="train", variable="grad_norm")
        .rename(columns={"grad_norm": "value"})
    )
    joint_df = pd.concat([train_loss_df, eval_loss_df, grad_norm_df])

    sns.set_context("talk")

    fg = sns.relplot(
        data=joint_df.iloc[1:],
        x="epoch",
        y="value",
        row="variable",
        hue="series",
        marker="o",
        kind="line",
        facet_kws={"sharey": False},
        aspect=2,
    )
    for ax in fg.axes.flat:
        ax.grid(alpha=0.5)
    fg.figure.savefig("metrics.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
