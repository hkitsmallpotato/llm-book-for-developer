# Crash course on Transformer

## NLP, Inductive bias, RNN

- What is NLP
- Neural network is strong (universal approximation), but works much better in practise when there is *inductive bias*
- RNN, LSTM etc are early attempts to apply neural network technique to NLP, face the problem of memory decay and long dependency

## Attention Mechanism

- Pivotal "Attention is all you need" paper
- Solves problems with RNN with perfect recall and no decay
- Differentiable version of Information Retrieval (IR) based on cosine similarity
- Enable parallel training
- Downside: quadratic runtime

## Tokenizer and Embedding

- Byte Pair Encoding for tokenization - compression based on linguistic lexicon structure
- Embedding - word2vec, one hop encoding to dense encoding, learning an embedding by neural network
- CLiP based and cosine-similarity based embeddings
- logit interpretation of latent space
- Positional embeddings

## Transformer Architecture

- Encoder-decoder (BERT) vs decoder-only
- Attention + Feedforward
- Revision: ResNet, inductive bias and compositionality, vanishing gradient problem


# Basics of LLM

## Scaling Law

- From BERT to GPT
- Chinhilla paper
- Double descent vs overtraining
- Data, compute, params

## Language Modeling, Text completion, sampling algorithms, and text generation

- Markov Chain as a naive form of language modeling
- Causality and masks
- seq2seq and types of training objective (next token prediction)
- Temperature, Beam search, top-p/k, nuclues sampling
- Mirostat

## GPT, Foundation models, and fine-tunes

- What are their differences

## Interpretations of LLM

- Autoregression and language model
- World modelling
- Knowledge engine
- Reasoning engine
- Universal information processing engine

## Ways to modify a model

- Prompt engineering, prefix tuning, lora, training


# Emergent abilities and prompt engineering

## In-context and few/one/zero shot learning

- Classes of emergent abilities
- Aquiring new skills, and embedded gradient descent

## Basics of prompt engineering

- What to do?

## Common prompting technique

- Chain of thoughts (CoT), ReACT, Reflexion


# Deploying Open source LLM

## Significant milestones in Open source LLM

- llama leak
- Alpaca paper
- ggml and llama.cpp
- Quantizations
- Vicuna

## Choosing a Model - family and technical spec

- Major families
- Param, tokens seen, (un)censored, context length limit, quantization method and level
- How to choose in practise

## Deployment ecosystems - HF, ggml, GPTQ

- HuggingFace, accelerate, bitsandbytes, FlashAttention
- Building llama.cpp
- Running on GPTQ
- Triton etc

## Bindings and API

- auto-gptq, llama-cpp-python
- OpenAI compatible API

## Frontends

- ooba


# LLM-enabled Application Development Frameworks

## Prompt Chaining, Tool using, Retrieval (and other) Augmentation

- Ideas of prompt chaining
- Toolformer and how LLM may use tools
- Overcoming LLM weaknesses through tool use augmentation

## Vector database

- Embedding and semantic search
- Chroma

## Types of LLM applications

- LUI
- Example/case studies: Ask documents, augmented chatbot, personal assistent, roleplaying

## Rapid Prototyping GUI

- Quick review of gradio
- Layout
- Actions and callback
- API and client

## llama-index


## langchain




# Agentization

## From tools to agents

- Tool AI wants to be agents AI
- Goal and internal mental states

## Cognitive Architecture

- Goal
- Todo list
- Memory
- Critic

## Examples: AutoGPT, BabyAGI

## Guidiance and other problems

- Format-conformance through low level manipulation
- Stuck in a loop

## Simulacra and CAMAL

- Deeper meaning of role-playing
- Society of communicating agents - cooperation to solve harder tasks


# Future of LLM

## Gap between Open source and propiertrary offers

- Leaderboards and metrics/evaluations
- Exegerated claims?
- Raw intelligence vs style imitation

## Context length limit and the quadratic attention bottleneck

- Why the context length is so short
- Dual problems

## Multi-modality

- Embodied intelligence hypothesis
- Mini-GPT4

## Coding ability and other weak spots




# Fine-tuning and training LLM



# LLMOps

# Securities




