# Layer Diffuse Refiners

Transparent Image Layer Diffusion using Latent Transparency for [refiners](https://github.com/finegrain-ai/refiners/tree/main).

![layerdiffuse archi](./assets/archi.png)

**Note:** This repo only implements layer diffusion for foreground generation base on the [Transparent Image Layer Diffusion using Latent Transparency](https://arxiv.org/abs/2402.17113v3) paper. Click [here](https://github.com/layerdiffusion/sd-forge-layerdiffuse) for the official implementation.

## Get Started:

> Clone this repo

```console
git clone https://github.com/chloedia/layerdiffuse.git
cd layerdiffuse
```

> Set up the environment using [rye](https://rye-up.com/)

```console
rye sync --all-features \
source .venv/bin/activate
```

And you are ready to go ! You can start by launching the generation of a cute panda wizard by simply running :

```console
python3 src/layerdiffuse/inference.py --checkpoint_path "checkpoints/" --prompt "a futuristic magical panda with a purple glow, cyberpunk" --save_path "outputs/"
```
**Note**: It should take time the first time as it will download all the necessary weight files. If you want to use different SDXL checkpoints you can use the refiners library to convert those file.
> Install your own sdxl weight files using the convertion script of refiners (see [refiners](https://github.com/finegrain-ai/refiners/tree/main) docs).

Go check into the refiners [docs](https://refine.rs/guides/adapting_sdxl/#multiple-loras) and especially the part to add loras or Adapters on top of the layer diffuse, create the assets for any of your creations that matches a specific style.

## Example of outputs

![panda generation](./assets/panda.png)
![glass generation](./assets/glass.png)

## What's Next ?

> Add style aligned to the generated content, to align a batch to the same style with or without a ref image;

> Add post processing for higher details quality (hands);


Don't hesitate to contribute! 🔆

---

Thanks to @limiteinductive for his help toward this implementation !
