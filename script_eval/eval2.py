import torch
from tqdm import tqdm
import argparse
import sys 
sys.path.append("..")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='patchfm', choices=['patchfm', 'tirex'], help='Model to evaluate')
    parser.add_argument('--dataset', type=str, default='artificial', choices=['artificial', 'utsd'], help='Dataset to evaluate on')
    parser.add_argument('--context_length', type=int, default=1024, choices=[32, 64, 128, 256, 512, 768, 1024], help='Context length for evaluation')

    args = parser.parse_args()
    BASE_MODEL = args.model
    CONTEXT_LENGTH = args.context_length

    print(f"Evaluating model: {BASE_MODEL} on context length: {CONTEXT_LENGTH}")

    if args.dataset == "artificial":
        print("Using artificial dataset")
        dataset = torch.load("../data/full.pt")

    elif args.dataset == "utsd":
        from dataset import UTSDataset
        print("Using UTSD dataset")
        dataset = UTSDataset(input_len=CONTEXT_LENGTH, output_len=32, flag="val")

    testloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    if BASE_MODEL == "patchfm":

        from forecaster import Forecaster, PatchFMConfig

        model_name = "huge_v3.pth"

        config = PatchFMConfig()
        config.ckpt_path = f"../ckpts/{model_name}"
        if "huge" in config.ckpt_path:
            config.n_heads = 64
            config.d_model = 2048

        model = Forecaster(config)

        all_losses = []
        for input, target in tqdm(testloader):
            input = input[:, -CONTEXT_LENGTH:]
            out, _ = model(input, quantiles=[0.5])
            loss = (out.detach().cpu() - target)**2
            loss = loss.mean(dim=1)
            all_losses.append(loss)
        all_losses = torch.cat(all_losses, dim=0)
        print(f"Mean loss for {model_name} on {args.dataset} dataset for context length {CONTEXT_LENGTH} : {all_losses.mean().item()}") 

    if BASE_MODEL == "tirex":

        from tirex import load_model, ForecastModel

        model: ForecastModel = load_model("NX-AI/TiRex")
        model.eval()

        all_losses = []
        for input, target in tqdm(testloader):
            input = input[:, -CONTEXT_LENGTH:]
            with torch.no_grad():
                quantiles, out = model.forecast(context=input, prediction_length=32)
            loss = (out.detach().cpu() - target)**2
            loss = loss.mean(dim=1)
            all_losses.append(loss)
        all_losses = torch.cat(all_losses, dim=0)
        print(f"Mean loss for TiRex on {args.dataset} dataset for context length {CONTEXT_LENGTH} : {all_losses.mean().item()}")


# save tmux logs : tmux capture-pane -S - -E - \; save-buffer ./tmux_buffer.log \; delete-buffer