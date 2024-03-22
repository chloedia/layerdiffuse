# layerdiffuse
Implementation of layer diffuse inference using refiners

For the Layer diffuse weights, you need to dl :

``` 
curl -L 'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors' --output "vae_transparent_decoder.safetensors" \

curl -L 'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors' --output "vae_transparent_encoder.safetensors"

curl -L "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors" --output "layer_xl_transparent_conv.safetensors"

curl -L 'https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors' --output "layer_xl_transparent_attn.safetensors"
```
Then you need to convert the encoder and the decoder weights:

```
python3 src/convertor/convert_transparent_diffusion.py --unet-from "vae_transparent_decoder.safetensors" --offset-from "vae_transparent_decoder.safetensors" --verbose --half
```

You also need to dl the weights for sdxl.





