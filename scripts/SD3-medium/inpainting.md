| Feature | **DEADiff** | **SD3‑Medium** |
| --- | --- | --- |
| Inpainting mechanism | DDIM latent overwrite | Dedicated inpainting ControlNet |
| Mask usage | Freezes latent pixels | Mask is a conditioning channel |
| ControlNet role | Only canny/depth guidance | Inpainting + canny + depth |
| Semantic awareness | None (mask is not “seen”) | Full semantic mask awareness |
| Architecture | SD1‑style UNet | SD3 transformer + patch VAE |
| Result | Good for preserving regions, limited regeneration quality | High‑quality, structure‑aware inpainting |