# Fine-Tuning Llama-3.2-1B with Axolotl and QLoRA

This repository contains a Jupyter Notebook for fine-tuning the [Llama-3.2-1B](https://huggingface.co/NousResearch/Llama-3.2-1B) model using the Axolotl framework and QLoRA (Quantized Low-Rank Adaptation). The notebook is designed to run in the Google Colab environment with T4 GPU support.

## Overview

In this project, we utilize the Axolotl library to simplify the process of fine-tuning large language models. By leveraging QLoRA, we can efficiently adapt the LLaMA model to new tasks with reduced memory requirements, making it ideal for training on limited hardware resources.

### Key Features

- **Fine-tuning**: Fine-tune the Llama-3.2-1B model.
- **Dataset**: Use the [story summarization dataset](https://huggingface.co/datasets/nldemo/story-summarization-demo) from Hugging Face.
- **Efficiency**: Implement efficient training techniques with QLoRA.
- **Ease of Use**: Utilize Axolotl for easy configuration and model training.

## Requirements

Before running the notebook, ensure that you have access to Google Colab with GPU support. The recommended GPU for this project is the T4 GPU.

### Install Dependencies

The necessary dependencies are automatically installed within the notebook, including:

- **Axolotl**: A framework for fine-tuning Llama models.
- **Flash Attention**: An optimized attention mechanism for faster training.
- **DeepSpeed**: A library for efficient distributed training.

## Getting Started

1. **Clone the Repository**

   Clone this repository in Google Colab:

   ```bash
   !git clone https://github.com/umairrkhn/Finetuning-Llama3-using-Axolotl.git
   ```

   Note: Make sure you are connected to the T4GPU runtime

2. **Navigate to the Directory**

    Change your working directory to the cloned repository:
  
    ```bash
    %cd Finetuning-Llama3-using-Axolotl
    ```

## Configurations

The training configuration is defined in a YAML file within the notebook and includes the following key elements:

- **Base Model**: 
  - Selection of the model to be used: `NousResearch/Llama-3.2-1B`.
  
- **Dataset Path**: 
  - The path to the dataset: `nldemo/story-summarization-demo`. 
  - Refer to the [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/) to learn more about different dataset formats.
  
- **Adapter Type**: 
  - Utilization of `qlora` for low-rank adaptation.
  
- **Hyperparameters**: 
  - Configuration of various training parameters, including:
    - Adapter (QLora)
    - Learning rate
    - Batch size
    - Gradient accumulation settings
    - Number of Epochs
    - Optimizer

For more detailed information on the available configuration options, please refer to the [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/docs/config.html).

## Model and Dataset Links

- **Base Model**: [Llama-3.2-1B](https://huggingface.co/NousResearch/Llama-3.2-1B)
- **Dataset**: [Story Summarization Dataset](https://huggingface.co/datasets/nldemo/story-summarization-demo)
- **Axolotl Repository**: [Axolotl on GitHub](https://github.com/axolotl-ai-cloud/axolotl)

## Acknowledgments

- **Axolotl**: For providing an excellent framework for model fine-tuning.
- **Hugging Face**: For hosting the model and datasets.
- **Google Colab**: For providing a free and easy-to-use platform for running Jupyter notebooks with GPU support.
