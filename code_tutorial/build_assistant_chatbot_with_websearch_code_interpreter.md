# Build an assistant chatbot with Web search and Code Interpreter

In this end to end tutorial, we will try to recreate a (relatively) full featured assistant chatbot that are equipped with web search and the code interpreter feature.

## Design and project setup

Our project will have these dependencies:

`requirements.txt`:
```
llama-cpp-python
outlines
transformers
huggingface_hub
duckduckgo-search
chromadb
llama-index-core
llama-index-vector-stores-chroma
llama-index-embeddings-huggingface
llama-index-readers-web
docker
```

Notice that we will be using `llama-index` as astarting point for the RAG part. It requires installing both the core library as well as appropiate plugins/addons for each specific integration we want. In the future we may opt to use more sophisticated/well designed RAG library.

Also note the library to interface with duck duck go web search (`duckduckgo-search`) and with `docker` for code interpreter.


## Basic Chatbot with Function calling skeleton

Let's develop iteratively. In the first round we will only have a barebone chatbot, but the main prompt should already have the skaffolding provided to implement function calling so that we can plug things in in the next rounds.

**LLM Setup**:

Nothing new, just redoing what we've been doing before using `llama-cpp-python` (See previous tutorials for details):

`infra/llm_provider.py`
```py
from llama_cpp import Llama, LlamaDiskCache

LOCAL_PATH = "/home/zeus/.cache/huggingface/hub/models--mradermacher--bagel-8b-v1.0-GGUF/snapshots/2b34e5c62aa0af4fb2dbc9a120e78ee357d0b026/bagel-8b-v1.0.Q6_K.gguf"


llm = Llama(model_path=LOCAL_PATH, n_ctx=8192)
llm.set_cache(LlamaDiskCache(capacity_bytes=(2 << 30)))

def stream_completion_outputs(llm_stream):
    response = ""
    for t in llm_stream:
        u = t["choices"][0]["text"]
        response = response + u
        print(u, end='', flush=True)
        #break
    return response
```

Perhaps the one new thing is we now wrap the logic for printing a streaming output live on the terminal into a function for convinience.

**Setting up the main prompt**

Basically, a bunch of text prompts implemented as pythong f-string and then substituted. But for some added convinience, we use the `outlines` library instead (but we don't use its actual core feature which is to do structured generation - instead we're only using its utility for prompt templating).

`core/prompt_templates.py`:
```py
import outlines

@outlines.prompt
def system_inst(sys, funcs, eg):
    """{{ sys }}
    Available functions:
    {% for f in funcs %}
    {{ f | source }}
    {% endfor %}
    {{ eg }}
    """

@outlines.prompt
def chatml_template(messages, nudge_role = "assistant"):
    """
    {% for m in messages %}
    <|im_start|>{{ m['role'] }}
    {{ m['content'] }}
    <|im_end|>
    {% endfor %}
    <|im_start|>{{ nudge_role }}
    """

system_prompt = \
"""As an AI assistant, you will answer user queries. When necessary, you can call the \
following functions to help you. Answer to user are done by sending a message to the \
user, identifying yourself as the assistant role via `<|im_start|>assistant`, \
while function calls are done by sending a special message to the function role implicitly \
via `<|im_start|>function` (Here the purpose is not to identify yourself, but to indicate \
that the message is for internal use). When sending to the function role, use \
the JSON format: a JSON object with attr "function" (For the function name) and "params" (a \
JSON object whose keys are the argument name, and the respective value are the correspdoning \
value of that argument). Function call return value (if any) will \
be sent to you as a message \
from system role, with further instruction on what you should do next also given there.

You are an open source AI assistant made by an annoymous person. You are cheerful, creative, \
but also thoughtful.
"""

eg = """Examples:
User> hi, how are you?
Correct role choice: assistant

User> What are some latest news in <topic>?
Correct role choice: function

User> Can you generate a graph of the SP500 today?
Correct role choice: function

User> Suggest some interior design for my new home.
Correct role choice: assistant
"""


fn_schema = """
{
    "title": "FunctionCall",
    "type": "object",
    "properties": {
        "function": { "type": "string", "enum": ["web_search_synthesis", "code_interpreter"] },
        "params": { "type": "object" }
    },
    "required": ["function", "params"]
}
"""
```

At the lowest level, we'll be using the `chatml` format, which is well suited for function calling and similar agent like use case. This is as shown in `chatml_template` above.

One trick and detail in the chatml template is that we have a `nudge_role`. This trick is about switching to use the low level text completion directly, but insert texts that are supposedly the LLM's response. That is, we "put words into the AI's mouth". The reason for this is that sometimes a LLM may go off-rail, so if we force the beginning part of AI response, it will be more likely to go where we want. In this tutorial, the main use of `nudge_role` is to preset a specific role in AI's reply.

Next, the overall organization of our system prompt is in `system_inst`. It consists of the basic instruction `system_prompt`, followed by a listing of the available functions for it to call, followed by few shot examples `eg`. Note that we again use the `outlines` utility to use meta-programming to auto-extract function source code from the actual function we define in python later.

Finally, `fn_schema` is for us to actually enforce the constraint of the LLM output later on.

> Note: Here we manually describe the required JSON output format, which is found to work better for weak models (This program is written a few months ago). With more recent, stronger model, directly injecting the JSONSchema into the prompt might be better.

**Main Program with Function Calling Constraint**:

Finally, we wire things up. We will also need to implement the constrainted output. Here we opt to do it manually:

`/chatbot.py`:
```py
from llama_cpp import Llama, LlamaDiskCache, LlamaGrammar, LlamaTokenizer

import json

from infra.llm_provider import llm, stream_completion_outputs

from core.prompt_templates import *
#from core.function_call import *

web_question1 = "What is the latest news on the war in Ukraine? I'm interested in the impact on financial markets."
offline_question1 = "Hi there! How's your day going?"

init_sys = { "role": "system", "content": system_inst(system_prompt, [], eg) }

message_history = [init_sys]

tokenizer = LlamaTokenizer(llm)
function_token = str(tokenizer.encode("function", add_bos=False)[0])
assistant_token = str(tokenizer.encode("assistant", add_bos=False)[0])

fn_call_schema = LlamaGrammar.from_json_schema(fn_schema)
```

Let's look at it more closely.

In the first line we do the import for `llama-cpp-python`, and we use much of the advanced features it have too - please refer to the Bonus/Extra tutorial for explanations.

We set the initial message list in OpenAI API format: `init_sys` is the formatted system message object using prompt template `system_init` - note that the function list is empty now. Then `message_history` list initially have just the system message.

To prepare for structured output, we do two things:

1. Choosing the reply role as either `assistant` (Direct answer) or `function` (Function calling mode): As this is an either-or choice (or more generally, a "one out of finitely many" choice), we can use simple logit bias. But to do this we need to get the token ID.
2. The function calling schema is converted to `llama-cpp-python` format as previous tutorial did.

Let's continue to the main loop:
```py
while True:
    user_q = input("User> ")
    message_history.append({ "role": "user", "content": user_q })
    p1 = chatml_template(message_history, nudge_role = "")
    o1 = llm(prompt=p1, max_tokens=1, logit_bias={ function_token: 50.0, assistant_token: 50.0 })
    role_pick = o1["choices"][0]["text"]
    print("Role picked: " + role_pick)
    if role_pick == "function":
        o2 = llm(prompt=p1 + "function\n", max_tokens=500, grammar=fn_call_schema, stop=["<|im_end|>"])
        print(o2)
        message_history.append({ "role": "function", "content": o2["choices"][0]["text"] })
        fn_obj = json.loads(o2["choices"][0]["text"])
        args = fn_obj["params"]
        fn_result = function_registry[fn_obj["function"]](**args)
        #message_history.append({ "role": "system", "content": "We are in development mode. Functions are not yet implemented, so please simulate the result as if the function is called successfully."})
        message_history.append({ "role": "system", "content": f"{fn_result['further_instruct']}\n\n# Function call result:\n{fn_result['answer']}"})
        p2 = chatml_template(message_history, nudge_role="assistant")
        o2_stream = llm(prompt=p2, max_tokens=500, stop=["<|im_end|>"], stream=True)
        response = stream_completion_outputs(o2_stream)
        message_history.append({ "role": "assistant", "content": response })
    elif role_pick == "assistant":
        llm_stream = llm(prompt = p1 + "assistant\n", max_tokens=800, stop=["<|im_end|>"], stream=True)
        response = stream_completion_outputs(llm_stream)
        message_history.append({ "role": "assistant", "content": response })
    else:
        raise ValueError("Unknown role:" + role_pick)
```

Wow, that's quite a lot, let's break it down again.

```py
    user_q = input("User> ")
    message_history.append({ "role": "user", "content": user_q })
```

Take user input and append to message list.

```py
    p1 = chatml_template(message_history, nudge_role = "")
    o1 = llm(prompt=p1, max_tokens=1, logit_bias={ function_token: 50.0, assistant_token: 50.0 })
    role_pick = o1["choices"][0]["text"]
    print("Role picked: " + role_pick)
```

Do the choice constrained LLM generation using `logit_bias`.

```py
    if role_pick == "function":
        o2 = llm(prompt=p1 + "function\n", max_tokens=500, grammar=fn_call_schema, stop=["<|im_end|>"])
        print(o2)
        message_history.append({ "role": "function", "content": o2["choices"][0]["text"] })
        fn_obj = json.loads(o2["choices"][0]["text"])
```

In the case of function calling (i.e. if LLM picked the function role), we do constrained generation again, this time using the grammar feature of `llama-cpp-python`, and then parse the JSON.

```py
        args = fn_obj["params"]
        fn_result = function_registry[fn_obj["function"]](**args)
        #message_history.append({ "role": "system", "content": "We are in development mode. Functions are not yet implemented, so please simulate the result as if the function is called successfully."})
        message_history.append({ "role": "system", "content": f"{fn_result['further_instruct']}\n\n# Function call result:\n{fn_result['answer']}"})
```

We haven't implemented this part yet, but basically we'll extract manually the function call's function name as well as the arguments, then use `function_registry` to dynamically call the function, using python's argument splicing to inject the arguments dynamically.

We will inject the result of function call into the prompt/message history.

```py
...
    elif role_pick == "assistant":
        llm_stream = llm(prompt = p1 + "assistant\n", max_tokens=800, stop=["<|im_end|>"], stream=True)
        response = stream_completion_outputs(llm_stream)
        message_history.append({ "role": "assistant", "content": response })
    else:
        raise ValueError("Unknown role:" + role_pick)
```

In the case of direct answer (or answer generation after function call), we nudge the role to assistant, then do generation as usual with streaming output.

## Code Interpreter using Docker Sandbox

Next, let's add a code interpreter. In a code interpreter, LLM is given interactive console access to a container sandbox, through which it will iteratively execute commands to try to accomplish a goal.

**Programmatically control a docker container**:

First we need to actually implement the tool itself.

`infra/docker_container_session.py`:

```py
from pathlib import Path
import docker

from dataclasses import dataclass

loop_command = "/bin/bash -c -- 'while true; do sleep 30; done;'"

@dataclass(frozen=True)
class DockerContainerConfig:
    image: str = "python:3.12"
    bind_dir: str = "/usr/src/app"
```

Some basic config and setups.

```py
class DockerCodeInterpreterSession:
    def __init__(self, storage_dir : str):
        self.client = docker.from_env()
        self.base_dir = Path(storage_dir)
    
    def _prepare_drive(self, session_name : str):
        mount_dir = self.base_dir.joinpath(session_name)
        mount_dir.mkdir(parents=True, exist_ok=True)
        return mount_dir
    
    def upload_files(self, session_name : str, files):
        my_dir = self._prepare_drive(session_name)
        for file in files:
            with open(my_dir.joinpath(file["path"]), "w") as f:
                f.write(file["content"])
```

We implement the tool as a class. Before actually running a container, we want to be able to work with the filesystem binding. Note a few things:

- The Docker SDK initialized as `docker.from_env()`
- Use of the official `pathlib` library for working with path. We also did `mkdir`.
- Then write file as usual for file uploads.

```py
    def start_session(self, session_name : str, container_config : DockerContainerConfig):
        # First prepare the drive
        mount_dir = self._prepare_drive(session_name)
        # Then setup the vol mapping
        vol_map = {}
        vol_map[str(mount_dir.absolute())] = {
            "bind": container_config.bind_dir,
            "mode": "rw"
        }
        # Finally start it
        cont = self.client.containers.run(container_config.image,
            detach=True,
            name=session_name,
            volumes=vol_map,
            command=loop_command)
        self.container = cont
        self.session_name = session_name
        self.default_work_dir = container_config.bind_dir
```

To start a code interpreter session, we will first setup the drive and then do a volumne mapping. The data structure to specify the volumne mapping is according to the requirement of `docker` SDK library (?). Here `str(mount_dir.absolute())` is the drive of the host/supervisor, while `container_config.bind_dir` is the guest/path inside the container.

Next, we actually use SDK to programmatically start running a container (`self.client.containers.run()`). `detach=True` coupled with the infinite loop sleep command is necessary to keep it running in the background otherwise it will immediately exits.

```py
    def run_single_command(self, command : str, work_dir = None):
        if work_dir is None:
            my_work_dir = self.default_work_dir
        else:
            my_work_dir = work_dir
        result = self.container.exec_run(command, workdir=my_work_dir)
        return result.output.decode("utf-8")

    def stop_session(self):
        self.container.stop()
```

Finally, to send a command to a running container, use `self.container.exec_run()`. Outputs from terminal is then extracted and returned.

**ReACT Prompt for code interpreter and execution loop**:

> TODO

