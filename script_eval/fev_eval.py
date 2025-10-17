import time

import datasets
import pandas as pd
import torch

from patchfm import Forecaster, PatchFMConfig

import fev

datasets.disable_progress_bars()

def batchify(lst: list, batch_size: int = 32, max_context_length: int = 1024):
    """Convert list into batches of desired size.
    If some elements have incompatible shapes, yield them individually.
    """
    for i in range(0, len(lst), batch_size):
        batch = lst[i : i + batch_size]
        batch = [x[-max_context_length:] for x in batch]

        try:
            # Try to stack into one tensor of shape [B, T, ...]
            batch_tensor = torch.stack(batch)
            yield batch_tensor
        except RuntimeError:
            # If stacking fails (different lengths or shapes), yield one by one
            print("Warning: yielding batch elements one by one due to incompatible shapes.")
            for x in batch:
                yield x.unsqueeze(0)  # keep batch dimension = 1

def predict_with_model(
    model,
    task: fev.Task,
    max_context_length: int = 1024,
    batch_size: int = 1,
) -> tuple[list[datasets.DatasetDict], float, dict]:

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("torch").cast_column("target", datasets.Sequence(datasets.Value("float32")))
        loaded_targets = past_data["target"]
        
        start_time = time.monotonic()

        all_preds, all_quantiles = [], []
        for batch in batchify(loaded_targets, batch_size=batch_size, max_context_length=max_context_length):
            
            pred, quantiles = model(
                batch, quantiles=task.quantile_levels, forecast_horizon=task.horizon
            )
            all_preds.append(pred.cpu())
            all_quantiles.append(quantiles.cpu())
        pred = torch.cat(all_preds, dim=0).numpy()
        quantiles = torch.cat(all_quantiles, dim=0).numpy()

        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": pred}
        for idx, level in enumerate(task.quantile_levels):
            predictions_dict[str(level)] = quantiles[:, :, idx]

        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    return predictions_per_window, inference_time, {}


if __name__ == "__main__":
    model_name = "PatchFM"
    num_tasks = None  # replace with `num_tasks = None` to run on all tasks

    config = PatchFMConfig(compile=True, load_from_hub=True)
    model = Forecaster(config)

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(model, task)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name="patchfm",
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        #print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv("patchfm.csv", index=False)