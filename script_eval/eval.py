import torch
from tqdm import tqdm
from einops import rearrange

from model import PatchFMLit

if __name__ == "__main__":

    dataset = torch.load("../data/full.pt")
    testloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

    model_name = "patchfm_200_80wu4_256.ckpt"

    model = PatchFMLit.load_from_checkpoint(f"../ckpts/{model_name}")
    model.eval()

    all_losses = []
    for input, target in tqdm(testloader):
        with torch.no_grad():
            out = model.model(input.to('cuda'))[..., list(model.model.quantiles).index(0.5)]
        x_patch = rearrange(input, "b (pn pl) -> b pn pl", pl=model.model.patch_len)[:, 1:, :]
        target = torch.cat((x_patch, target.unsqueeze(1)), dim=1)
        loss = (out.detach().cpu() - target)**2
        loss = loss.view(loss.shape[0]*loss.shape[1], -1).mean(dim=1)
        all_losses.append(loss)
    all_losses = torch.cat(all_losses, dim=0)
    print(f"Mean loss for {model_name} : {all_losses.mean().item()}")

# loss 80 epochs only warmups with 4 warmup steps only artificial batch_size 256 old Revin : 102541.5703125
# timeoss lightweight 80 epochs only warmups with 4 warmup steps only artificial batch_size 256 : 900000.000

# total loss 200 ; 80 epochs warmups with 4 steps artificial batch_size 256 new Revin : 72502.3828125 