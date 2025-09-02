import torch
from tqdm import tqdm
import argparse
import sys 
sys.path.append("..")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='patchfm', choices=['patchfm', 'tirex'], help='Model to evaluate')
    parser.add_argument('--context_length', type=int, default=1024, choices=[32, 64, 128, 256, 512, 768, 1024], help='Context length for evaluation')

    args = parser.parse_args()
    BASE_MODEL = args.model
    CONTEXT_LENGTH = args.context_length

    print(f"Evaluating model: {BASE_MODEL} on context length: {CONTEXT_LENGTH}")

    dataset = torch.load("../data/full.pt")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    if BASE_MODEL == "patchfm":

        from model import PatchFMLit

        model_name = "patchfm_200_80wu4_256.ckpt"

        model = PatchFMLit.load_from_checkpoint(f"../ckpts/{model_name}")
        model.eval()

        all_losses = []
        for input, target in tqdm(testloader):
            input = input[:, -CONTEXT_LENGTH:]
            with torch.no_grad():
                out = model.model(input.to('cuda'))[:, -1, :, list(model.model.quantiles).index(0.5)]
            loss = (out.detach().cpu() - target)**2
            loss = loss.mean(dim=1)
            all_losses.append(loss)
        all_losses = torch.cat(all_losses, dim=0)
        print(f"Mean loss for {model_name} on context length {CONTEXT_LENGTH} : {all_losses.mean().item()}") 

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
        print(f"Mean loss for TiRex on context length {CONTEXT_LENGTH} : {all_losses.mean().item()}")


# save tmux logs : tmux capture-pane -S - -E - \; save-buffer ./tmux_buffer.log \; delete-buffer