# train.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.encoder import Encoder
from models.decoder import Decoder
from utils.payload import make_random_string_batch
from utils.metrics import psnr, ber_from_logits
from utils.bits import bits_to_string, threshold_logits_to_bits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Hyperparams
    L = 256          # message bits
    n_chars = 8     # random ASCII chars per sample
    batch_size = 128
    epochs = 15   # было 3
    lr = 3e-4   # было 2e-4

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Models
    enc = Encoder(L=L, hidden=96).to(device)
    dec = Decoder(L=L, hidden=128).to(device)

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    loss_img = nn.MSELoss()
    loss_msg = nn.BCEWithLogitsLoss()

    beta = 12.0
    alpha = 1.0

    enc.train()
    dec.train()

    for epoch in range(1, epochs + 1):
        running_psnr = 0.0
        running_ber = 0.0
        n_steps = 0

        for x, _ in train_loader:
            x = x.to(device)  # [B,3,32,32]

            m_bits = torch.randint(0, 2, (x.size(0), L), device=device).float()            
            texts = None

            x_stego, delta = enc(x, m_bits, return_delta=True)
            logits = dec(x_stego)

            Limg = loss_img(x_stego, x)
            Lmsg = loss_msg(logits, m_bits)
            lambda_delta = 0.02  # старт
            Ldelta = delta.abs().mean()
            loss = alpha * Limg + beta * Lmsg + lambda_delta * Ldelta

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                running_psnr += psnr(x, x_stego)
                running_ber += ber_from_logits(logits, m_bits)
                n_steps += 1

            # small demo print sometimes
            if n_steps % 200 == 0:
                with torch.no_grad():
                    demo_bits = threshold_logits_to_bits(logits[0]).float()
                    demo_ber = (demo_bits != m_bits[0]).float().mean().item()

                decoded = bits_to_string(demo_bits)

                print(f"[epoch {epoch} step {n_steps}] "
                    f"loss={loss.item():.4f} "
                    f"PSNR={running_psnr/n_steps:.2f} "
                    f"BER={running_ber/n_steps:.4f}")

                print(f"[epoch {epoch} step {n_steps}] loss={loss.item():.4f} PSNR={running_psnr/n_steps:.2f} BER={running_ber/n_steps:.4f}")

                demo_bits = threshold_logits_to_bits(logits[0]).float()
                demo_ber = (demo_bits != m_bits[0]).float().mean().item()
                print("  demo BER     :", f"{demo_ber:.4f}")

        print(f"Epoch {epoch}: PSNR={running_psnr/n_steps:.2f} BER={running_ber/n_steps:.4f}")

    # Save
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(), "L": L}, "stego_ed_cifar10.pt")
    print("Saved: stego_ed_cifar10.pt")


if __name__ == "__main__":
    main()
