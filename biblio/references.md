# Ch 1 Attention and Transformer

Modern NLP, LLM, and the ChatGPT revolution is possible all thanks to the attention mechanism and transformer architecture, aside from Moore's law and scaling up neural network training. This all begins with the landmark "Attention is all you need" paper. Although future architecture may change, having a good understanding of these core mechanisms give you the foundation to work effectively with LLM. The references below contain both visual explanations and code focused ones. It is also interesting to note theoretical attempts at re-interpreting, and hence understanding attention mechanism, from new, insightful angle.

At the same time, while applying attention to construct the Transformer based, LLM neural network architecture, we must also have supportive elements around it - this includes the Tokenizers, token and positional embeddings. The references below explore these concepts in more depth.


BPE Tokenization Demystified: Implementation and Examples, by MartinLwx. https://martinlwx.github.io/en/the-bpe-tokenizer/

What Are Word and Sentence Embeddings? Cohere LLM University, Chapter 1, Module 1. By Luis Serrano. https://cohere.com/llmu/sentence-word-embeddings

A Deep Dive into NLP Tokenization and Encoding with Word and Sentence Embeddings, by Josh Pause. https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/

Word Embedding Analogies: Understanding King - Man + Woman = Queen, by Kawin Ethayarajh. https://kawine.github.io/blog/nlp/2019/06/21/word-analogies.html

Linda Linsefors's Shortform, lesswrong forum. https://www.lesswrong.com/posts/tM84DyBg4Jbq5zGmH/linda-linsefors-s-shortform?commentId=owWTRrnGDfEqyGzjb
(See also https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85 , referenced from above)

Rotary Embeddings: A Relative Revolution, Biderman, Stella and Black, Sid and Foster, etc; Eleuther blog. https://blog.eleuther.ai/rotary-embeddings/

Chapter 1.3 Rotary Positional Embeddings, The Large Language Model Playbook, Cyril Zakka. https://cyrilzakka.github.io/llm-playbook/nested/rot-pos-embed.html

Annotated Research Paper Implementations, Rotary Positional Embeddings (RoPE), by labml. https://nn.labml.ai/transformers/rope/index.html

Tutorial/implementation of GPT, by labml. https://nn.labml.ai/transformers/gpt/index.html

The Illustrated Transformer, by Jay Alammar. https://jalammar.github.io/illustrated-transformer/

The Illustrated GPT-2 (Visualizing Transformer Language Models), by Jay Alammar. http://jalammar.github.io/illustrated-gpt2/

Attention is Kernel Trick Reloaded, Gokhan Egri and Xinran (Nicole) Han. https://egrigokhan.github.io/data/cs_229_br_Project_Report_KernelAttention.pdf

# Ch 2 Training LLM and Text Generation

As Moore's law continue and compute capacity increase, a new paradigm emerged that suggests training large scale neural network with a generic, instead of task specific objective. (TODO)

In the huggingface reference below, various sampling algorithms for LLM are discussed. Sampling algorithm are used to turn LLM from a "next token predictor" into a text generator, by repeatedly sampling from it in auto-regressive mode. It is interesting to observe that in practise, there is a certain art, or "black magic", to using sampling algorithm to improve LLM output quality. The open source community have made some interesting contributions in this regard.

The true nature of LLM is a hotly debated topic. Generally speaking, there are two "camps" - those who believe they are merely modelling surface statistics and does not posses true reasoning ability nor intelligence, and those who are believe they are something more. Here, we take a pragmatic position: as long as scientific research into this matter produces new way to interpret and conceptualize LLM, we may consider adding them to our arsenal of tools. Later on, as we develop LLM based GenAI applications, these understandings and "ways of thinking" may help us design and program these LLM better. With this being said, let's take a tour of some common ideas.

On the camp that suggests LLM may have acquired some degree of intelligence, some perspective includes:

- World modelling - LLM maintain an internal representation of the world as a state variable, that is sequentially updated as it reads the text input.
- Grokking and double descent - challenging the traditional notion of overfitting and bias variance trade-off, it suggests that it is a developmentally intermediate phase, and when neural networks are further trained at a massive scale, eventually the neurons collase into a "deep understanding" that recognize the underlying pattern and interpolate the training data in a robust manner.
- Reasoning and knowledge engine
- A cognitive computer - while classical computer operates on bits as data and uses classical, boolean logic; LLM can be conceptualized as a new form of computer that takes on natural language text as both data and "natural language program".

Reference pending: Chinhilla scaling law paper

Reference pending: The foundation model paradigm, the pre-training/fine-tuning/preference optimization phase.

How to generate text: using different decoding methods for language generation with Transformers, by Patrick von Platen. https://huggingface.co/blog/how-to-generate

ChatGPT is a blurry JPEG of the web, by Ted Chiang. https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web

ChatGPT Is Not a Blurry JPEG of the Web. It's a Simulacrum. By Domenic Denicola. https://blog.domenic.me/chatgpt-simulacrum/

Think of language models like ChatGPT as a "calculator for words", by Simon Willison. https://simonwillison.net/2023/Apr/2/calculator-for-words/

Do Large Language Models learn world models or just surface statistics? By Kenneth Li. (Note: Talks about Othello-GPT and world modelling) https://thegradient.pub/othello/

Understanding grokking in terms of representation learning dynamics, by Eric J. Michaud. https://ericjmichaud.com/grokking-squared/

Deep Networks Always Grok and Here is Why, ICML 2024. By Ahmed Imtiaz Humayun, Randall Balestriero, and Richard Baraniuk. https://imtiazhumayun.github.io/grokking/

Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. Alethea Power, Yuri Burda, Harri Edwards, etc. https://arxiv.org/abs/2201.02177

Explaining grokking through circuit efficiency. Vikrant Varma, Rohin Shah, Zachary Kenton, etc. https://arxiv.org/abs/2309.02390

LLMs are Not Just Next Token Predictors. Stephen M. Downes, Patrick Forber, Alex Grzankowski. https://arxiv.org/abs/2408.04666

Double Descent - Part 1: A Visual Introduction. By Jared Wilber & Brent Werness. https://mlu-explain.github.io/double-descent/

Double Descent Demystified - Identifying, Interpreting & Ablating the Sources of a Deep Learning Puzzle. By Rylan Schaeffer, Zachary Robertson, Akhilan Boopathy, etc. https://iclr-blogposts.github.io/2024/blog/double-descent-demystified/

Scaffolded LLMs as natural language computers, by Beren Millidge. https://www.beren.io/2023-04-11-Scaffolded-LLMs-natural-language-computers/ (See also discussion thread at https://www.lesswrong.com/posts/43C3igfmMrE9Qoyfe/scaffolded-llms-as-natural-language-computers )

Reference pending: Massively multitask perspective

# Ch 3 Emergence, In Context Learning, and Prompt Engineering

https://thegradient.pub/in-context-learning-in-context/

https://ai.stanford.edu/blog/understanding-incontext/

https://www.lakera.ai/blog/what-is-in-context-learning

https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/

https://www.jasonwei.net/blog/emergence

https://cset.georgetown.edu/article/emergent-abilities-in-large-language-models-an-explainer/


https://www.promptingguide.ai/

https://learnprompting.org/docs/basics/prompt_engineering

https://learnprompting.org/docs/basics/formalizing



# Ch 4 Running Open source LLM






https://a16z.com/emerging-architectures-for-llm-applications/

https://arxiv.org/abs/2307.09793

https://llmmodels.org/

https://llm.extractum.io/






https://osanseviero.github.io/hackerllama/blog/posts/hitchhiker_guide/

https://www.redditmedia.com/r/LocalLLaMA/comments/1atghbb/local_llm_glossary_simple_llama_sillytavern_setup/

https://interconnected.org/home/2024/07/19/ai-landscape



https://hackernoon.com/efficient-guided-generation-for-large-language-models-llm-sampling-and-guided-generation
https://arxiv.org/abs/2307.09702

https://simmering.dev/blog/structured_output/

https://github.com/sgl-project/sglang

https://uptodata.substack.com/p/guided-generation-for-llm-outputs



https://thenewstack.io/a-comprehensive-guide-to-function-calling-in-llms/

https://thenewstack.io/building-an-open-llm-app-using-hermes-2-pro-deployed-locally/

https://ai.google.dev/gemini-api/docs/function-calling

https://e2b.dev/blog/how-to-add-code-interpreter-to-llama3
https://dev.to/tereza_tizkova/llama-3-with-function-calling-and-code-interpreter-55nb


----

# Others

https://nuvalence.io/insights/a-6-category-taxonomy-for-generative-ai-use-cases/

https://towardsai.net/p/artificial-intelligence/generative-ai-terminology-an-evolving-taxonomy-to-get-you-started

----

# Second round

https://thegabmeister.com/blog/run-open-llm-local/

https://vercel.com/guides/openai-function-calling


https://sumanthrh.com/post/distributed-and-efficient-finetuning/

https://kaitchup.substack.com/p/a-guide-on-hyperparameters-and-training

https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft

https://docs.adapterhub.ml/methods.html


https://graphcore-research.github.io/galore/
https://medium.com/@tanalpha-aditya/galore-memory-efficient-llm-training-by-gradient-low-rank-projection-d93390e110fe


https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/


https://insujang.github.io/2024-01-07/llm-inference-autoregressive-generation-and-attention-kv-cache/
https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/
https://insujang.github.io/2024-01-21/flash-attention/


https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8


https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120

https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in
https://predibase.com/blog/lora-exchange-lorax-serve-100s-of-fine-tuned-llms-for-the-cost-of-one


https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html




https://kai-greshake.de/posts/llm-malware/

https://kai-greshake.de/posts/in-escalating-order-of-stupidity/


https://medium.com/emalpha/safeguarding-llm-conversations-using-llama-guard-a1652da1d2de

https://blog.langchain.dev/rebuff/
https://www.guardrailsai.com/docs/concepts/guard



https://lmsys.org/blog/2024-02-05-compressed-fsm/

https://blog.dottxt.co/how-fast-cfg.html

https://lmsys.org/blog/2024-01-17-sglang/


https://pub.towardsai.net/autonomous-gpt-4-from-chatgpt-to-autogpt-agentgpt-babyagi-hugginggpt-and-beyond-9871ceabd69e
https://sea.mashable.com/tech/23298/auto-gpt-babyagi-and-agentgpt-how-to-use-ai-agents


https://lilianweng.github.io/posts/2023-06-23-agent/
https://medium.com/the-modern-scientist/a-complete-guide-to-llms-based-autonomous-agents-part-i-69515c016792


https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/

https://www.width.ai/post/llm-powered-autonomous-agents

https://blog.langchain.dev/what-is-a-cognitive-architecture/

https://github.com/ysymyth/awesome-language-agents


https://sajalsharma.com/posts/overview-multi-agent-fameworks/
https://wandb.ai/vincenttu/blog_posts/reports/Exploring-2-Multi-Agent-LLM-Libraries-Camel-Langroid--Vmlldzo1MzAyODM5


https://arxiv.org/abs/2402.01680
https://arxiv.org/abs/2304.03442

https://arxiv.org/abs/2305.16291


https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html
http://kubeagi.k8s.com.cn/docs/Configuration/DistributedInference/deploy-using-rary-serve/




