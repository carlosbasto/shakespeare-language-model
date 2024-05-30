# Shakespeare Language Model with SAP AI Core

## Welcome to the "SAP AI Core is All You Need" Series!

We're thrilled to have you join us on this exciting journey into the world of AI and language models using SAP AI Core and SAP AI Launchpad. This repository contains all the code needed for our series, guiding you through the powerful capabilities of SAP AI Core to build, deploy, and manage your own AI models with ease.

## What to Expect from the Series

In this series, we'll cover a wide range of topics to equip you with the knowledge and skills needed to leverage SAP AI Core for your AI projects. Here’s an overview of what we will develop through this series:

### 1. Building Your Own Language Model with Transformers
**Introduction**: Dive into the world of Transformers, the architecture behind models like GPT. We’ll guide you through building a language model from scratch, focusing on the Shakespearean language.

- **Key Concepts**:
  - Understanding Transformers: Explore the architecture and attention mechanisms.
  - Implementing Transformers from Scratch: Learn to create a language model tailored to specific needs.

[SAP AI Core is All You Need | 1. Building Your Own Language Model with Transformers](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-1-building-your-own-language-model-with/ba-p/13687781)
[ai-core/ai-core-training](./ai-core/ai-core-training)

### 2. Setting the Stage for a Shakespeare-Language Model
**Introduction**: Ensure everything is operational for deploying a Shakespearean Language Model.

- **Key Concepts**:
  - Setting Up SAP AI Core and Launchpad: Configure the environment for deployment.
  - Defining Resource Groups: Manage resources and workloads.
  - Configuring GitHub Repositories: Store and manage workflow definitions.
  - Creating Docker Registry and Object Store Secrets: Manage execution outputs and input files.

[SAP AI Core is All You Need | 2. Setting the Stage for a Shakespeare-Language Model](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-2-setting-the-stage-for-a-shakespeare-language/ba-p/13689458)
[ai-core/ai-core-training-setup](./ai-core/ai-core-training-setup)
[ai-core/ai-core-training](./ai-core/ai-core-training)
[ai-core/ai-core-datasets](./ai-core/ai-core-datasets)

### 3. Workflow, Configuration, and Shakespeare Language Model Training
**Introduction**: Learn to containerize and orchestrate AI text generation pipelines using Docker and Argo Workflows.

- **Key Concepts**:
  - Creating Docker Images: Set up and build Docker images for the training pipeline.
  - Designing Workflow Templates: Automate and manage the training process.
  - Deploying the Training Workflow: Deploy the workflow template on SAP AI Core.

[SAP AI Core is All You Need | 3. Workflow, Configuration, and Shakespeare Language Model Training](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-3-workflow-configuration-and-shakespeare/ba-p/13689844)
[ai-core/ai-core-training-setup](./ai-core/ai-core-training-setup)
[ai-core/ai-core-training](./ai-core/ai-core-training)
[ai-core/templates](./ai-core/templates)

### 4. Improving Model Training Efficiency with Checkpointing/Resuming
**Introduction**: Dive into checkpointing, a technique to save and resume training efficiently.

- **Key Concepts**:
  - Understanding Checkpointing: Learn its importance and implementation.
  - Creating Separate Docker Images: Enhance modular design and scalability.
  - Adapting Code for Checkpointing: Modify code to support checkpointing.
  - Deploying the Checkpointer Workflow: Manage the checkpointing process effectively.

[SAP AI Core is All You Need | 4. Improving Model Training Efficiency with Checkpointing/Resuming](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-4-improving-model-training-efficiency-with/ba-p/13690949)
[ai-core/ai-core-checkpointer-setup](./ai-core/ai-core-checkpointer-setup)
[ai-core/ai-core-checkpointer](./ai-core/ai-core-checkpointer)
[ai-core/templates](./ai-core/templates)

### 5. Fine-Tuning with Low-Rank Adaptation (LoRA)
**Introduction**: Explore fine-tuning pre-trained models to excel in specific tasks.

- **Key Concepts**:
  - Understanding Fine-Tuning: Learn its significance in machine learning.
  - Defining the Fine-Tuning Task: Focus on language style transfer to Shakespearean English.
  - Implementing LoRA: Step-by-step guide using PyTorch.
  - Deploying the Fine-Tuning Workflow: Deploy the fine-tuning workflow using SAP AI Core.

[SAP AI Core is All You Need | 5. Fine-Tuning with Low-Rank Adaptation (LoRA)](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-5-fine-tuning-with-low-rank-adaptation-lora/ba-p/13694357)
[ai-core/ai-core-fine-tuner-setup](./ai-core/ai-core-fine-tuner-setup)
[ai-core/ai-core-fine-tuner](./ai-core/ai-core-fine-tuner)
[ai-core/templates](./ai-core/templates)
[ai-core/ai-core-datasets](./ai-core/ai-core-datasets)

### 6. Serving Shakespeare Model using SAP AI Core and KServe
**Introduction**: Deploy and serve AI models using SAP AI Core and KServe, focusing on the Shakespeare Language Model.

- **Key Concepts**:
  - Deploying AI Models: Integrate custom classes and modules.
  - Building a Text Generation API: Set up a Flask app for generating Shakespearean text.
  - Logging in MLOps: Monitor and troubleshoot the deployment process.

[SAP AI Core is All You Need | 6. Serving Shakespeare Model using SAP AI Core and KServe](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-6-serving-shakespeare-model-using-sap-ai-core/ba-p/13696608)
[ai-core/ai-core-generator](./ai-core/ai-core-generator)
[ai-core/templates](./ai-core/templates)

### 7. Deploying Language Models for Text Generation
**Introduction**: Cover serving templates, Docker builders, and deployment to bring the models to life.

- **Key Concepts**:
  - KServe Serving Template: Design and implement a serving template in KServe.
  - Multi-Stage Docker Builds: Create efficient Docker images.
  - Deploying with SAP AI Core: Detailed guide to model deployment.

[SAP AI Core is All You Need | 7. Deploying Language Models for Text Generation](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-7-deploying-language-models-for-text-generation/ba-p/13712187)
[ai-core/ai-core-generator](./ai-core/ai-core-generator)
[ai-core/ai-core-generator-tst](./ai-core/ai-core-generator-tst)
[ai-core/templates](./ai-core/templates)

### 8. Consuming and Sampling from Shakespeare Language Models
**Introduction**: Deploy and consume the fine-tuned models for Shakespeare-style text transfer.

- **Key Concepts**:
  - Deploying the Fine-Tuned Model: Step-by-step deployment guide.
  - Creating an Application: Build a user-friendly application to interact with the models.

[SAP AI Core is All You Need | 8. Consuming and Sampling from Shakespeare Language Models](https://community.sap.com/t5/technology-blogs-by-sap/sap-ai-core-is-all-you-need-8-consuming-and-sampling-from-shakespeare/ba-p/13708364)
[ai-core/ai-core-app](./ai-core/ai-core-app)

## Ready to Dive In?

We're excited to embark on this journey with you. By the end of this series, you’ll have a comprehensive understanding of building, fine-tuning, deploying, and consuming AI models using SAP AI Core. Stay tuned and get ready to unlock the potential of SAP AI Core and Launchpad!

For all the code needed for this series, including Docker images, Python scripts, templates, and more, please explore this repository.

Let's get started!
