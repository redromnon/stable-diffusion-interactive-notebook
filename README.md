<div align="center"> <h1> Stable Diffusion Interactive Notebook 📓 🤖 </h1> 

 <a target="_blank" href="https://colab.research.google.com/github/redromnon/stable-diffusion-interactive-notebook/blob/main/stable_diffusion_interactive_notebook.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 
 
 _Click the above button to start generating images!_
</div>

<br/>

A widgets-based interactive notebook for Google Colab that lets users generate AI images from prompts (Text2Image) using [Stable Diffusion (by Stability AI, Runway & CompVis)](https://en.wikipedia.org/wiki/Stable_Diffusion). 

This notebook aims to be an alternative to WebUIs while offering a simple and lightweight GUI for anyone to get started with Stable Diffusion.

Uses Stable Diffusion, [HuggingFace](https://huggingface.co/) Diffusers and [Jupyter widgets](https://github.com/jupyter-widgets/ipywidgets).

![GUI screenshot](https://github.com/redromnon/stable-diffusion-interactive-notebook/assets/74495920/461b23dc-ea92-4f11-b3a2-0593f51e2c43)

## Features
- Interactive GUI interface
- Available Stable Diffusion Models:
  - [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - [Stable Diffusion 2.1 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
  - [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
  - [OpenJourney v4](https://huggingface.co/prompthero/openjourney-v4)
  - [Dreamlike Photoreal 2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0)
- Available Schedulers:
  - [EulerAncestralDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers/euler_ancestral)
  - [EulerDiscreteScheduler](https://huggingface.co/docs/diffusers/api/schedulers/euler)
  - [DDIMScheduler](https://huggingface.co/docs/diffusers/api/schedulers/ddim)
  - [UniPCMultistepScheduler](https://huggingface.co/docs/diffusers/api/schedulers/unipc)
- Includes Safety Checker to enable/disable inappropriate content
- Features [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse) Autoencoder for producing "smoother" images
