# LLM Applications Notebooks

This repository includes a series of notebooks that detail initial experiments with Large Language Models (LLMs) using specific datasets and frameworks under the constraints of limited computational resources. The notebooks are designed to share insights and experiences from these experiments.

## Technologies Used

- **Hugging Face Transformers:** A state-of-the-art library providing a vast range of machine learning models primarily focused on Natural Language Processing (NLP). Transformers facilitate easy access to pre-trained models which can be used for tasks like text classification, information extraction, question answering, and more.

- **LangChain:** A library designed to enhance the capabilities of language models by enabling them to interact more effectively within applications. LangChain provides tools to integrate conversational AI into various environments, making it easier to build applications that require advanced language understanding and generation.

- **PyTorch:** An open-source machine learning library widely used for applications in artificial intelligence, including deep learning. PyTorch offers dynamic computation graphs that allow for flexibility and speed in model development and training.


## Notebooks Description

1. **Fine-tuning with LoRA:**
   - **Dataset:** The `dialogsum` dataset from Hugging Face Datasets, designed for dialogue summarization. It provides a challenging testbed for fine-tuning language models to understand and summarize conversational content. [DialogSum Dataset](https://huggingface.co/datasets/knkarthick/dialogsum)
   - This notebook demonstrates the process of fine-tuning a language model using the LoRA technique. The focus is on adapting the model to better handle summarization tasks, despite hardware limitations.

2. **RAG for BioASQ:**
   - **Dataset:** The `mini-bioasq` dataset, a subset of the BioASQ challenge focused on biomedical question answering. This dataset helps in testing the Retrieval-Augmented Generation (RAG) model in a domain-specific scenario. [Mini-BioASQ Dataset](https://huggingface.co/datasets/rag-datasets/mini-bioasq)
   - This experiment explores the implementation of RAG for biomedical question answering, constrained by computational capabilities.

## Hardware Limitations

The experiments in this repository are significantly impacted by the available hardware resources, which limit the ability to utilize larger models or perform extensive fine-tuning. These limitations particularly hinder the potential to perform computationally intensive tasks like full model training and extensive inference processes.

## Future Directions

- **Agent-Based Systems:** Plans include expanding into projects that develop language-based agents using LangChain, aiming to improve interaction capabilities and contextual understanding in complex scenarios.
- **Model Expansion:** As resources improve, there is an intention to explore more sophisticated and larger models to fully leverage the capabilities offered by advanced NLP technologies.




