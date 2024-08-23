# Getting Started with Local LLM for developer

Developer have different needs from end user and are more comfortable with getting technical, so here's a quickstart guide specifically for developer.

## Download Model

Generally, most models are downloaded from Huggingface. So first locate the model's *main* repo (that contains the unquantized, huggingface `transformers` compatible model files). What happen next depends on which LLM inference backend you use:

- For llama.cpp (recommended for beginners), you need to find the corresponding GGUF quantized repo. In this case you do not download the whole repo, but just one single `.gguf` file with the right quantization level for you (this depends on the memory vs intelligence tradeoff - generally q4 is a reasonable default, but many repo will have a guideline in their README telling you how to choose also).
- For exllama v2, find the exl repo, then download the full repo, **BUT** need to choose the correct git branch for selecting the quantization level instead.

Once you've identified it, there are some recommended methods:

### Method 1: Using the huggingface_hub library

> Reference: https://huggingface.co/docs/huggingface_hub/v0.21.4/guides/cli#huggingface-cli-download

Advantage: can handle gated repo (those that requires you to request access) by first logging in via `huggingface_hub login`

First install the library by `pip3 install huggingface_hub`

Then, to download a repo: `huggingface_hub download <repo id> [list of individual files] --revision <branch/tag/commit hash>`

It will be saved to the huggingface cache, by default at `$HOME/.cache/huggingface`. The repos downloaded uses a special symlink structure to allow downloading multiple versions with dedup. (But on Window, symlink is not supported so it will be expanded out)

Notice that you can use use this programmatically, which can be useful for infra tooling that want to download models on behave of user.

### Method 2: Using the aria2c command line tool

This can only handle a single file easily (so GGUF only), on the other hand, it has the advantage of having good options for things like concurrent connections, resuming download, more visibility into the progress and details, etc.

After install, run the command `aria2c -c -d <download location> -x <num of connections> [--disable-ipv6] <Download URL>`

To find the actual download link for a file, in Huggingface website, click on the file (git LFS), which will show the details, then click "copy download link".

## Run Model

> Reference: https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md

Here we will only go through doing this for llama.cpp.

First install it. If you want to compile yourself, follow the instruction on README. This usually boils down to a `make` but with additional env variables and preparations needed depending on what hardware acceleration option you want to use.

Otherwise, you may download pre-compiled binaries on the github release page. Make sure to find the right combo of OS and hardware acceleration supported. For example avx2 is for pure CPU (reasonably modern CPU), openblas for CPU but with acceleration for the input prompt processing part, cuda for Nvidia GPU (make sure the version match what you have on your computer).

The simplest way to get started is to run an interactive chat. Here's an example with Nvidia GPU:

```
.\llama-b3497-cuda12\llama-cli.exe -m .\LLM_Model_Collections\meta-llama-3.1-8b-instruct-abliterated.Q3_K_M.gguf -cnv --chat-template llama3 -ngl 99 -c 40000 -fa -ctk q4_0 -ctv q4_0 -mli
```

There are quite a few command line options:

- The basic is `-m` for model file path and `-c` for context length (below or up to the model's supported context length)
- Then `-cnv --chat-template llama3` specifies to have an interactive chat, using the `llama3` prompt template
- `-mli` is for quality-of-life as it allows multiline input for user in console
- `-ngl` specify how many layers of the model to offload to GPU

Next is the special option:
- `-fa -ctk q4_0 -ctv q4_0` enable flash attention, and then ask that both the K part and V part of the KV-cache is quantized to `q4_0` level. This is important when operating at longer context.

Now some explanation for the prompt template. Except for base model, LLM are trained to usually expect a specific prompt format, where usually there will be some special tokens delineating user/assistant/system messages. Deviating from it may results in degraded intelligence (this may be more severe in older model than more recent ones). (Some models may also be trained in multiple prompt template to be more robust to this) This information about what prompt template to use is now sometimes included as metadata in the GGUF file itself, but specifying it yourself is more sure. Note that only a preset list of common template is allowed, see the list at https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template Now if you want to use custom one, you will have to follow instruction there.

Alternatively, you can use the trio of `-r "User:" --in-prefix " " --in-suffix "Assistant:"`, which with some creative programming might be able to reproduce the effect of a custom template.

Finally, you can also run it as a server. The options are similar, but the executable is now `llama-server`.
