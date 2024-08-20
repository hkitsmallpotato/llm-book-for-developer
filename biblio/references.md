# Ch 1 Attention and Transformer

Modern NLP, LLM, and the ChatGPT revolution is possible all thanks to the attention mechanism and transformer architecture, aside from Moore's law and scaling up neural network training. This all begins with the landmark "Attention is all you need" paper. Although future architecture may change, having a good understanding of these core mechanisms give you the foundation to work effectively with LLM. The references below contain both visual explanations and code focused ones. It is also interesting to note theoretical attempts at re-interpreting, and hence understanding attention mechanism, from new, insightful angle.

At the same time, while applying attention to construct the Transformer based, LLM neural network architecture, we must also have supportive elements around it - this includes the Tokenizers, token and positional embeddings. The references below explore these concepts in more depth.

## Tokenizer, embeddings and positional embeddings

BPE Tokenization Demystified: Implementation and Examples, by MartinLwx. https://martinlwx.github.io/en/the-bpe-tokenizer/

What Are Word and Sentence Embeddings? Cohere LLM University, Chapter 1, Module 1. By Luis Serrano. https://cohere.com/llmu/sentence-word-embeddings

A Deep Dive into NLP Tokenization and Encoding with Word and Sentence Embeddings, by Josh Pause. https://datajenius.com/2022/03/13/a-deep-dive-into-nlp-tokenization-encoding-word-embeddings-sentence-embeddings-word2vec-bert/

Word Embedding Analogies: Understanding King - Man + Woman = Queen, by Kawin Ethayarajh. https://kawine.github.io/blog/nlp/2019/06/21/word-analogies.html

Linda Linsefors's Shortform, lesswrong forum. https://www.lesswrong.com/posts/tM84DyBg4Jbq5zGmH/linda-linsefors-s-shortform?commentId=owWTRrnGDfEqyGzjb
(See also https://blog.esciencecenter.nl/king-man-woman-king-9a7fd2935a85 , referenced from above)

Rotary Embeddings: A Relative Revolution, Biderman, Stella and Black, Sid and Foster, etc; Eleuther blog. https://blog.eleuther.ai/rotary-embeddings/

Chapter 1.3 Rotary Positional Embeddings, The Large Language Model Playbook, Cyril Zakka. https://cyrilzakka.github.io/llm-playbook/nested/rot-pos-embed.html

Annotated Research Paper Implementations, Rotary Positional Embeddings (RoPE), by labml. https://nn.labml.ai/transformers/rope/index.html

## Attention and Transformer

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

## Foundation Models

Reference pending: Chinhilla scaling law paper

Reference pending: The foundation model paradigm, the pre-training/fine-tuning/preference optimization phase.

## LLM Sampling Algorithms

How to generate text: using different decoding methods for language generation with Transformers, by Patrick von Platen. https://huggingface.co/blog/how-to-generate

## Perspectives on LLM

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

Prompt engineering arguably have its ancestral root before the ChatGPT revolution, when NLP researchers probing earlier forms of Transformer based model discovered emergent phenomenon coming out of those models. As model scale increases, they may suddenly acquire new capability for new tasks. Whether emergence is actually real or just a mirage is a topic of debate, though it does suggests that 1. Rather than focusing on just training and evaluation in traditional ML, the way we give instructions to it (i.e. prompting) may have effects on how the model performs, and that 2. there are things we don't fully understand about these models that warrant further investigation.

One of the most significant candidate for emergent ability, that is arguably foundational to prompt engineering, is what is known as "In Context Learning" (ICL). The first reference on ICL below gives a detailed and academic review, and examined two proposed theoretical explanations - Bayesian conditioning, and mesa-optimization. While the two may seem opposed on first glance, it can be seem in some way that both are pieces to the puzzle of ICL.

Then, we provide two resources for prompt engineering - Prompt Engineering Guide by DAIR.AI is more academic and have a good list of the common special prompting technique, such as Chain of thought and ReACT, that will be a neccesary ingredient for building LLM based app in future. On the other hand, the free source by "Learn Prompting" is more oriented towards layperson, but the section on "Formalizing Prompt" is a good summary of the basic prompting technique that should be applied regardless.

## Emergent abilities

Emergent Abilities of Large Language Models, by Ryan O'Connor. https://www.assemblyai.com/blog/emergent-abilities-of-large-language-models/

137 emergent abilities of large language models, By Jason Wei. https://www.jasonwei.net/blog/emergence

Emergent Abilities in Large Language Models: An Explainer, by Thomas Woodside. https://cset.georgetown.edu/article/emergent-abilities-in-large-language-models-an-explainer/

## In Context Learning

In-Context Learning, In Context. By Daniel Bashir. https://thegradient.pub/in-context-learning-in-context/

How does in-context learning work? A framework for understanding the differences from traditional supervised learning, by Sang Michael Xie and Sewon Min. https://ai.stanford.edu/blog/understanding-incontext/

What is In-context Learning, and how does it work: The Beginner’s Guide, by Deval Shah. https://www.lakera.ai/blog/what-is-in-context-learning

## Prompt Engineering

Prompt Engineering Guide, produced by DAIR.AI (Democratizing Artificial Intelligence Research, Education, and Technologies). https://www.promptingguide.ai/

Introductory Course on Generative AI and Prompt Engineering, written by Sander Schulhoff, organized by "Learn Prompting". https://learnprompting.org/docs/introduction

Formalizing Prompts, by Sander Schulhoff. https://learnprompting.org/docs/basics/formalizing



# Ch 4 Running Open source/Open weight LLM

Especially for people who prefers hands on, running an open source/open weight LLM on their own computer is a nice way to get started with developing with LLM. But this haven't been always possible, and its development has short but significant history. The links below should provide some hints. To sum up:

- There are efforts trying to reproduce LLM pre-ChatGPT era (eg GPT-J)
- Modern open source/open weight LLM is reliant on release by large, well resourced institution/organization (kickstarted by llama1 from meta)
- Model extraction, distillation (Alpaca paper and more), synthetic datasets (Textbook is all you need/phi model by Microsoft), are techniques that can makes training strong LLM easier. There is some sort of boostrap effect here.
- Software ecosystem matters - llama.cpp have played a pivotal role in lowering the barrier of entry of doing LLM inferencing on consumer grade and even edge device. It started its life as a hack with tricks such as mmap added later, but has since matured into a core library in the ecosystem. Before llama.cpp, LLM are mostly run directly using a research/prototype oriented framework such as Huggingface Transformers, that assume a well resourced user with powerful computer. llama.cpp manually reimplement the inferencing (and later on, also some part of training) making it possible to apply various optimization to achieve the advantages listed above.
- Later on, as competition heats up, open source/open weight LLM become mainstream and eventually becomes a viable alternative to closed source API.

In the next section, some guides and resources for running LLM on your own device is given. Overall, knowing lots of terminology can be confusing at first so some of the guides below provide a glossary and explainer. In general, it comes down to: choose and download model weights, choose the inferencing backend (and optionally frontend) and install it, then run. For beginners, I recommend llama.cpp as it is the most flexible in terms of hardware requirements (can run on both GPU, partially on GPU, or purely on CPU), and have good quantization format that significantly lower the requirements with relatively transparent trade-off (eg Huggingface transformer also technically have quantization via bitsandbytes, but it is not as advanced as the quantization methods offered by llama.cpp).

Finally, since our goal is to developer LLM based GenAI app, the OpenAI API, which have been a de facto standard, is important to abstract away different backend and provide a uniform interface for the application developer (as well as framework). As a result many backend also supports operating in server mode, which will provide HTTP endpoint, usually (but not always) conforming to the OpenAI API standard (called "Open AI API compatible").

## Some historical context

The History of Open-Source LLMs: Imitation and Alignment (Part Three), by Cameron R. Wolfe. https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-imitation

Large language models are having their Stable Diffusion moment, by Simon Willison. https://simonwillison.net/2023/Mar/11/llama/

Google "We Have No Moat, And Neither Does OpenAI", leaked memo republished on SemiAnalysis. https://www.semianalysis.com/p/google-we-have-no-moat-and-neither

Optimizing $Model.cpp, by Matt Rickard. https://blog.matt-rickard.com/p/optimizing-modelcpp

Edge AI Just Got Faster, by Justine Tunney. https://justine.lol/mmap/

Hacknews discussion on GGML. https://news.ycombinator.com/item?id=36215651

Intro to GGML, by Carson Tang. https://carsontang.github.io/machine-learning/2023/06/06/ggml-101/

https://lmsys.org/blog/2023-03-30-vicuna/

https://crfm.stanford.edu/2023/03/13/alpaca.html

## Guide to run open source LLM

Running Open Large Language Models Locally, by TheGabmeister. https://thegabmeister.com/blog/run-open-llm-local/

All LLM Directory, by @johnrushx. https://llmmodels.org/

LLM Explorer. https://llm.extractum.io/

The Llama Hitchiking Guide to Local LLMs, Omar Sanseviero. https://osanseviero.github.io/hackerllama/blog/posts/hitchhiker_guide/

Local LLM Glossary" & "Simple Llama + SillyTavern Setup Guide, by kindacognizant. https://www.redditmedia.com/r/LocalLLaMA/comments/1atghbb/local_llm_glossary_simple_llama_sillytavern_setup/

On the Origin of LLMs: An Evolutionary Tree and Graph for 15,821 Large Language Models, by Sarah Gao, Andrew Kean Gao. https://arxiv.org/abs/2307.09793


# Ch 5 Intro to LLM based GenAI app

LLM based GenAI app is an emerging tech that is in its early day. We begin by reviewing some recent proposed architecture and classification scheme of these new class of app. Overall, while general principles of UI/UX design and software engineering still applies, these apps have their own consideration and characeristics too based on the current capability, limitations, and "quirks" of current generation LLM.

A first hurdle that we immediately encounter is that to enable tool use (see prompt engineering chapter), we need LLM to reliably produce structured output that strictly adhere to some formal format. (Sidenote: one insightful comment said that this can be considered a glue between the language layer and logic/control layer of the app) This is a significant issue, and numerous attempts have been made to solve this. It is only in very recent time that OpenAI finally supports structured output in their official API, and before that it comes down to a mismash of prompt engineering, asking it to retry/correct/repair, interleaved prompting, abusing OpenAI API's function calling feature, and early attempt at an algorithmic solution to gaurantee technical adherence to the format. The situation is messy and complex, and comes down to some factors: 1. Although algorithmic solutions to it are already known fairly early on, they requires tight integration with the LLM, which OpenAI's API didn't have back then (it does have logit bias, but is sufficiently limited that it won't work for this purpose). On the other hand, locally run LLM can support this, so this created a sort of divergence when the trend is to want to treat OpenAI's API as similar to the servlet/WSGI standard sitting right in the middle layer of the tech stack. Moreover, as there is also non-algorithmic approach that can work, this further created fragmentation in the ecosystem of libraries that aim to support this. 2. The incidental complexity of point 1 created a side effect that the more technical problem of further refining the core algorithm to be more efficient, and to handle real world complexity, such as specifically supporting JSONSchema efficiently alongside context free grammar (which is considered general enough to serve as the common base case), are harder to attend to. 3. Other technical difficulties that are simply epheramal and only due to an immature ecosystem overall. Nevertheless they might have been enough to slow adoption, which creates a negative feedback effect.

Overall however, I think we can be cautiously optimistic. As OpenAI API's finally added support for structured output, and as these early experience accumulate, this part of the ecosystem may see a refactor and consolidation phase, after which libraries will become more mature and comprehensive.

At any rate, now is a good time to learn more about the core algorithm and some of its optimization and engineering to make it production ready.

After that, we look at function calling and code interpreter, two value added feature that OpenAI's API provide. Given what you've learnt so far, it should be apparent that given access to an open weight model and appropiate libraries, it is possible to build that yourself. (Note: contingent on the model being either sufficiently intelligent, or having had these features explicitly trained into the model as its native capability) Nevertheless, they should be considered to be nearer the infrastructure layer - application developer should use them as a higher level abstraction.

## Architecture and UI/UX/Product

Emerging Architectures for LLM Applications, by Matt Bornstein and Rajko Radovanovic. https://a16z.com/emerging-architectures-for-llm-applications/

Mapping the landscape of gen-AI product user experience, by Matt Webb. https://interconnected.org/home/2024/07/19/ai-landscape

## Structured output/Guided/Constrained Generation

https://hackernoon.com/efficient-guided-generation-for-large-language-models-llm-sampling-and-guided-generation
https://arxiv.org/abs/2307.09702

https://simmering.dev/blog/structured_output/

https://github.com/sgl-project/sglang

https://uptodata.substack.com/p/guided-generation-for-llm-outputs


https://lmsys.org/blog/2024-02-05-compressed-fsm/

https://blog.dottxt.co/how-fast-cfg.html

https://lmsys.org/blog/2024-01-17-sglang/


## OpenAI API Standard - Function Calling and Code Interpreter

https://vercel.com/guides/openai-function-calling

https://thenewstack.io/a-comprehensive-guide-to-function-calling-in-llms/

https://thenewstack.io/building-an-open-llm-app-using-hermes-2-pro-deployed-locally/

https://ai.google.dev/gemini-api/docs/function-calling

https://e2b.dev/blog/how-to-add-code-interpreter-to-llama3
https://dev.to/tereza_tizkova/llama-3-with-function-calling-and-code-interpreter-55nb


# Ch 6 Retrieval Augmented Generation (RAG)

> TODO: RAG articles are a dime a dozen given how it's seen as being the best match with enterprise use case. Because of this it is a challenge to filter and select articles from such a vast pool.

https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html



# Ch 7 LLM based Agents

LLM based Agents are very early, cutting edge, speculative technology. Nevertheless, if they work, it'd unlock lots of potentials. As well, from a purely technical perspective, for much of the LLM application pattern and prompt techniques, when pushed to its limit, they can all be interpreted as some form of limited agents. (From an application point of view, agent might be considered the analog of "dynamic" style programming language)

The first section below provide references that elucidates the theory side of agents. Understanding cognitive architecture and how to design a set of prompts for agent can be a useful skill for building other LLM apps in general.

After that, we examine some of the ambituous LLM agent projects in the earliest days. Although they may not work very well now, they should be seen more as a vision of what might be possible in the near future. For actually deploying agents in your LLM based GenAI applications, consider libraries and frameworks that are specifically designed for developers, such as langroid, langgraph, crewAI, etc.

Finally, multi-agent is an extension of single agent. Multi-agent allow us to decompose complex problems, which reduces the intelligence requirement placed on the underlying LLM. Another crucial point is that multi-agent also produces new phenomenon independently of the consideration of capability - having multiple agent introduces communication, coordination, and sociology. This open ups the interesting new domain of social simulation.

## Autonomous agents and Cognitive Architecture

https://lilianweng.github.io/posts/2023-06-23-agent/
https://medium.com/the-modern-scientist/a-complete-guide-to-llms-based-autonomous-agents-part-i-69515c016792


https://yoheinakajima.com/task-driven-autonomous-agent-utilizing-gpt-4-pinecone-and-langchain-for-diverse-applications/

https://www.width.ai/post/llm-powered-autonomous-agents

https://blog.langchain.dev/what-is-a-cognitive-architecture/

https://github.com/ysymyth/awesome-language-agents

## Agent framework in real world

https://pub.towardsai.net/autonomous-gpt-4-from-chatgpt-to-autogpt-agentgpt-babyagi-hugginggpt-and-beyond-9871ceabd69e
https://sea.mashable.com/tech/23298/auto-gpt-babyagi-and-agentgpt-how-to-use-ai-agents


## Multi-agent system

https://sajalsharma.com/posts/overview-multi-agent-fameworks/
https://wandb.ai/vincenttu/blog_posts/reports/Exploring-2-Multi-Agent-LLM-Libraries-Camel-Langroid--Vmlldzo1MzAyODM5



# Ch 8 Production Inference Engine

When deploying LLM based GenAI app to production, we'd need a production grade inference engine. This has different consideration and engineering principles compared to ones intended for consumers in Ch 4 above. Namely, a production grade engine is designed for large concurrency inference, have specific performance metric to meet, and generally optimize for the hardware/performance ratio as the deployment scales up; while consumer engine generally only need to consider single user performance and is focused on reducing the hardware requirement. This leads to different engineering techniques.

In the first section, we focus on some core techniques that distinguishes a production grade engine - Continuous batching, Paged Attention, KV-cache, and parallelism. (To be fair, consumer engine does also have some of these features) We also look at analysing the transformer architecture to derive formulas to predict inference engine performance.

In the second section, we probe deeper into some extended features, such as speculative decoding, and LoRaX to serve multiple loras, which can be seen as limited forms of model variations, at scale efficiently.

Finally, we look at some hands on tutorial to work with these production engine.

## Core features

https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/


https://insujang.github.io/2024-01-07/llm-inference-autoregressive-generation-and-attention-kv-cache/

https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/

https://insujang.github.io/2024-01-21/flash-attention/


https://medium.com/@plienhar/llm-inference-series-3-kv-caching-unveiled-048152e461c8

## Extended features

https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120

https://predibase.com/blog/lorax-the-open-source-framework-for-serving-100s-of-fine-tuned-llms-in

https://predibase.com/blog/lora-exchange-lorax-serve-100s-of-fine-tuned-llms-for-the-cost-of-one

## Hands on

https://docs.ray.io/en/latest/serve/tutorials/vllm-example.html

http://kubeagi.k8s.com.cn/docs/Configuration/DistributedInference/deploy-using-rary-serve/



# Ch 9 LLM Security

The special nature of LLM bring with it a new class of security vulnerability. Some parallel can be made with well known, existing class of exploits, such as prompt injection vs cross site scripting (XSS)/SQL injection attack. Unfortunately, while XSS is generally a well solved problem, at current state, it is difficult to devise a bullet proof defense for prompt injection.

In the attack section we see some elaboration of these methods of attack, and see how the proliferation of LLM app may bring an unintended consequences of opening up new, creative ways to extend the basic attack method into something that's harder to defend against, and have more severe consequences.

In the defense section, we look at some LLM side models and software libraries to implement defensive measures, in a hands on manner.

## LLM attack vector and class of vulnerability

https://kai-greshake.de/posts/llm-malware/

https://kai-greshake.de/posts/in-escalating-order-of-stupidity/

## LLM security defence

https://medium.com/emalpha/safeguarding-llm-conversations-using-llama-guard-a1652da1d2de

https://blog.langchain.dev/rebuff/

https://www.guardrailsai.com/docs/concepts/guard


# Ch 10 Fine-tuning LLM

As we saw in Ch 2, LLM can be fine-tuned to adapt it for specific task, or to change its behavior. For industry application with specific requirements, fine-tuning can be a way to create a tailored model that performs better on the task. (However careful consideration and evaluation is necessary - it is not enough to just fire and forget)

The references below first go through some general theory of how training is done by the frameworks, the optimizations involved to speed up training, and explain the configuration parameters and hyperparameters. Then, we look at LoRa, which is a more light-weight (but somewhat limited) alternative to a full fine-tuning.

## General theory

https://sumanthrh.com/post/distributed-and-efficient-finetuning/

https://kaitchup.substack.com/p/a-guide-on-hyperparameters-and-training

## Loras

https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft

https://docs.adapterhub.ml/methods.html

## Bonus topics

https://graphcore-research.github.io/galore/

https://medium.com/@tanalpha-aditya/galore-memory-efficient-llm-training-by-gradient-low-rank-projection-d93390e110fe


----

# Others (pending processing)

https://nuvalence.io/insights/a-6-category-taxonomy-for-generative-ai-use-cases/

https://towardsai.net/p/artificial-intelligence/generative-ai-terminology-an-evolving-taxonomy-to-get-you-started







https://arxiv.org/abs/2402.01680
https://arxiv.org/abs/2304.03442

https://arxiv.org/abs/2305.16291
