# Extra - More about llama-cpp-python

The core value provided by this python binding at a binary/native level is that 1. it allows running the LLM *in-process*, which might be preferrable to web server (even if local) for less moving parts. 2. It opens up the possibility to use it to implement custom extension to how inference is done, such as adding new features or changing the way it works.

But let's look at some less well-documented features first.

## Using long context

Initialize the `llama_cpp.Llama` object with extra arguments:

```py
from llama_cpp import Llama, GGML_TYPE_Q4_0

llm = Llama(
    model_path="your file path here",
    n_ctx=55000,
    n_gpu_layers=33,
    flash_attn=True,
    type_k=GGML_TYPE_Q4_0,
    type_v=GGML_TYPE_Q4_0,
    n_batch=2048
)
```

## Tokenizing

```py
from llama_cpp import Llama, LlamaTokenizer

# Define llm = Llama(...)

tokenizer = LlamaTokenizer(llm)
enc = tokenizer.encode("Some text here", add_bos=False, special=False) # Array of tokens
print(enc)

dec = tokenizer.decode(enc, prev_tokens=None) #returns b-string
print(dec)
```

## Using a cache

```py
from llama_cpp import Llama, LlamaDiskCache

# Remember to define LOCAL_PATH

llm = Llama(model_path=LOCAL_PATH, n_ctx=8192)
llm.set_cache(LlamaDiskCache(capacity_bytes=(2 << 30)))
```

## Grammar (for structured output)

> Note that JSON mode (including with or without JSONSchema) and Function calling are both supported but are already well documented on their website.

One thing I really like about this library is how it has good all-round supports on things with structured output. In the most general case, you can ask it to conform to a CFG (Context free grammar).

```py
from llama_cpp import Llama, LlamaGrammar

# Define llm = Llama(...) as usual
# Define json_schema_str as a string of the JSONSchema

my_cfg = LlamaGrammar.from_json_schema(json_schema_str)

response = llm(... , grammar=my_cfg)
```

At an even lower level, you can use `LlamaGrammar.from_string()`, then input argument is a string of the CFG (refer to https://github.com/ggerganov/llama.cpp/tree/master/grammars for the syntax to specify a grammar)

## Server

The internals are not documented, but at `llama_cpp/server/app.py`, it ultimately uses the functions in the main repo to create a fastapi app object: `create_app(settings: Settings | None = None, server_settings: ServerSettings | None = None, model_settings: List[ModelSettings] | None = None,`

