# Build an assistant chatbot with Web search and Code Interpreter

In this end to end tutorial, we will try to recreate a (relatively) full featured assistant chatbot that are equipped with web search and the code interpreter feature.

> Reference: https://github.com/lemonteaa/llm-chatbot-tuto (WIP, this version written against https://github.com/lemonteaa/llm-chatbot-tuto/tree/v0.0.1-alpha)

## Design and project setup

> TODO: write about the overall design

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

Let's design the prompt for code interpreter.

`core/code_interpreter.py`:

> Note: the formating may be different from actual source code due to markdown formatting issue with escaping backtick.

```py
import outlines
from llama_cpp import LlamaGrammar

from infra.llm_provider import llm

from infra.docker_container_session import *

q = "Please generate a plot of the price of SP500 in the last week with hourly data, and save it to a file named sp500-plot.png."

react_grammar = """
thought-format ::= ( "Thought " iter ": " thought "\n" )
thought ::= [^\n]+
action-format ::= ( "Action " iter ": " action-type-branch "--")
action-type-branch ::= ( write-file-action | command-action | exit-action )

command-action ::= ( "Terminal command\n" "> " terminal-command "\n")
terminal-command ::= [^\n]+

exit-action ::= "Exit\n"

write-file-action ::= ( "Write files\n" (write-individual-file)+ "Finish.\n" )
write-individual-file ::= ( "File name: `" file-name-pattern "`\n" "File content:\n\`\`\`\n" file-content "\`\`\`\n" )
file-name-pattern ::= ("./" [^`]+)
file-content ::= [^`]*

root ::= (thought-format action-format)
"""

@outlines.prompt
def code_interpreter_prompt(request, homedir, history):
    """
BEGIN_INSTRUCTION
This is code interpreter. You, as an AI assistant, can design and execute python program on behalf of the user to accomplish various goals.
Setup:
- The python program will be executed in a sandboxed enviornment with python3.10 and pip3 already installed.
- Terminal access is also given, sudo apt-get is allowed. OS is ubuntu 20.
- Public internet access is allowed.
- Access to the user's home folder at `{{ homedir }}` (including creating subfolders etc) is allowed.
- In your program, anything printed to stdout will be shown to you, but no stdin access is given directly.
- Terminal output will also be printed to you.
- Access to the sandbox/code interpreter is interactive - you can perform one action at each step, and use the result/feedback to iterate and refine until you achieve your goal.

Instruction:
You should engage in a thought loop, explaining what you would do based on the current state of the code interpreter/results of previous actions. You will also pay attention to any technical details.
In that loop, you should repeat thought-action-observation:
- Thought is your internal monologue to decide on what to do next based on all available informations.
- Action is where you execute an action. Three types of actions are possible with their own format requirements:
  - "Write files" is where you will create new files by entering the text.
  - "Terminal command" is where you will execute a terminal command.
  - "Exit" is when you are done and want to leave code interpreter. (See below)
- Observation is where the system will return the result of your terminal command.

When you're accomplished your goal, choose the action type: "Exit".
You will then be exfiltrated by the system into another section where you are free to write the final answer, with access to your whole transcript.

Example format for reference:

Thought 1: I should install system dependencies.
Action 1: Terminal command
> sudo apt-get update && sudo apt install htop -y
--
Observation 1:
<transcript of terminal output>

Thought 2: I should now create the python program.
Action 2: Write files
File name: `./main.py`
File content:
\`\`\`
import json
print("Hello")
\`\`\`
File name: `./requirements.txt`
File content:
\`\`\`
json
\`\`\`
Finish.
--
Observation 2: Success.

Thought 3: Let's run the python program.
Action 3: Terminal command
> pip install -r requirements.txt && python main.py
--
Observation 3:
<transcript of terminal output>
...
Hello

Thought 4: We've successfully printed "Hello" to our screen.
Action 4: Exit
--

(Above is just a sample, feel free to be more elaborate in your internal monologue.)
END_INSTRUCTION

User query:
{{ request }}

Assistant:
{% for h in history %}
{{ h }}
{% endfor %}
    """
```

That's a lot even for just plain prompt engineering!

- `react_grammar` is a manually written Grammar spec for `llama-cpp-python` that enforce the ReACT thinking and acting loop.
- In the main prompt, we give the context/background, then specify instruction explaining the ReACT pattern requirement, and then give an example. Finally we inject the interaction history of the running code interpreter session.


And then the execution loop:

```py
def code_interpreter_loop(request, session_name):
    conf = DockerContainerConfig()
    try:
        codeInterpreter = DockerCodeInterpreterSession("../code-int-storages")
        history = []
        done = False
        codeInterpreter.start_session(session_name, conf)
        while not done:
            prompt = code_interpreter_prompt(request, conf.bind_dir, history)
            response = llm(prompt=prompt, stop=["--"], grammar=LlamaGrammar.from_string(react_grammar))
            parsed_response = parse_response(response["choices"][0]["text"])
            if parsed_response["action_type"] == "Exit":
                done = True
                print("Code interpreter DONE!")
            elif parsed_response["action_type"] == "Terminal Command":
                commandOutput = codeInterpreter.run_single_command(parsed_response["command"], work_dir=conf.bind_dir)
                parsed_response["observation"] = commandOutput
            elif parsed_response["action_type"] == "Write files":
                pass # TODO
            else:
                raise ValueError()
            history.append(parsed_response)
    except BaseException as e:
        print(e)
    finally:
        codeInterpreter.stop_session()
```

Here, `history` is about the commands previously executed by code interpreter and its subsequent terminal output. We use it to format the `prompt`, and then generate LLM response, stopping at `--` to detect the action step of the ReACT loop. After parsing response, we run the command, gather the output, and then append to `history`. This is repeated until LLM decides it is done.

**Wire up the function call**:

Finally, some ceremony:

`core/function_call.py`:

```py
from infra.llm_provider import llm

# Functions for LLM to use

# TODO: need to return dict with further instructions

def code_interpreter(request: str) -> str:
    """Initiate a code interpreter session to solve problem, or do things on the 
    user's behave, via running python program designed by an AI.

    request: A fully self contained text instruction on what the AI should
      accomplish within the code interpreter session. Example: "Generate a plot of
      the price of SP500 in the last week, and save it to the file sp_500_price.png."
    
    Return: Text remarks from the code interpreter AI.
    """
    run_code_interpreter(request)

def run_code_interpreter(r):
    print("run_code_interpreter") # TODO: implement code interpreter
    print(r)

function_registry = {
    #"web_search_synthesis": web_search_synthesis,
    "code_interpreter": code_interpreter
}
```

Our general pattern here is that `code_interpreter` is the function with almost empty code body but with full Doc-strings in natural language, so this is the one that will be auto-extracted by `outlines` to inject into the main prompt when listing available functions.

Then `run_code_interpreter` would be responsible for actually executing things.

Finally we also add a `function_registry` dict for dynamically calling functions.

## Web Search and RAG

Another main feature of an assistant chatbot is to search the web and generate answer based on that. It is a specific example of RAG (Retrieval Augmented Generation). We will have tutorial to cover RAG more later on, but now let's just cook something up quickly with `llama-index`.

At its core, RAG consist of:
1. Data abstraction for Documents and Nodes (section of a document)
2. A data ingestion pipeline
3. A vector database with embedding and storage mechanism
4. A query pipeline

We will skip 4 (as we build our custom pipeline), leverage `llama-index` convinience in 2, and try to setup 3 correctly (which can be tricky due to quirks of the library).

But first let's review 1. Ultimately, they can be considered as object as container for the text data, but with auto links to related nodes. They also accept setting user defined metadata.

**RAG Setup**:

`infra/web_search_rag.py`:

```py
# Sample RAG chatbot

from transformers import AutoTokenizer

import json
import re

from duckduckgo_search import DDGS

from llama_index.readers.web import SimpleWebPageReader

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.node_parser import TokenTextSplitter

def extract_action(text):
    cap_grp = re.search("```(json|js|javascript)?([\s\S]*)```", text).group(2)
    return json.loads(cap_grp)

def web_search(q):
    results = DDGS().text(q + " filetype:html", max_results=10)
    return results


#def gen_docs(search_results):
#    docs = SimpleWebPageReader(html_to_text=True).load_data([result['href'] for result in search_results])
#    for result, doc in zip(search_results, docs):
#        doc.metadata = { 'title': result['title'], 'href': result['href'] }
#    return docs

def gen_docs(search_results):
    #docs = SimpleWebPageReader(html_to_text=True).load_data([result['href'] for result in search_results])
    docs = []
    for result in search_results:
        try:
            doc = SimpleWebPageReader(html_to_text=True).load_data([ result['href'] ])[0]
            doc.metadata = { 'title': result['title'], 'href': result['href'] }
            docs.append(doc)
        except BaseException as e:
            print(e)
    return docs

def query_crawl(docs, query, top_k, embed_model, chroma_client):
    chroma_collection = chroma_client.create_collection("temp")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return nodes
```

The web search is easy: just do `DDGS().text(q + " filetype:html", max_results=10)`. Note the restriction to HTML file type as the simple web reader we use from `llama-index` cannot handle media type.

`gen_docs` uses the `SimpleWebPageReader` in `llama-index` to read and parse the webpage URL into markdown text. We also perform data mapping, especially of metadata, from `DDGS` into the `llama-index` Documents.

`query_crawl` is the main integration point where we tie everything together. `chroma` is used as the vector database. What is particularly tricky here is that `llama-index` defaults to using OpenAI GPT and have many other deep-seated defaults that are not easy to change if your use case is significantly different from what it assumes. This code is settled on only after some trial and error. The central object here is `VectorStoreIndex`, but we have to override the default settings right at object construction time. `StorageContext` is set to use the `ChromaVectorStore`, which is fed to the main index to override the default. In a similar vein, we will later on feed in the embed model to use otherwise it may call OpenAI instead.

(?) Notice that document splitting into nodes is done automatically by `VectorStoreIndex`.

The latter code:

```py
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
```

Is standard.

**Wiring up with Function Calling**:

> Note: again, triple backtick escaping, so may be different from actual source code

First, setup a new prompt for our custom query pipeline:

`core/prompt_templates.py`:

```py
@outlines.prompt
def grounded_generation_prompt(mynodes, search_query, topic_question):
    """
BEGININPUT
{% for node in mynodes %}
BEGINCONTEXT
id: {{ loop.index }}
title: {{ node.metadata["title"] }}
ENDCONTEXT
{{ node.text }}
{% endfor %}
ENDINPUT
BEGININSTRUCTION
<system>Answer user query based on provided information above. In your answer, cite the sources *inline* using a special tag. Example:
\`\`\`
It is known that current progress in AI is <cite:3>expected to accelerates by some experts</cite>, although other disagree.
\`\`\`
Where the number inside the tag is the source id. </system>
Topic question: {{ topic_question }}
(If topic question is empty, infer what the user want to ask based on the search query below and answer that instead:
Search query: {{ search_query }})
ENDINSTRUCTION
    """
```

This uses the retrieved nodes from our RAG module, and then ask the LLM to generate a *grounded response* with citations.

`core/function_call.py`:

```py
from infra.web_search_rag import *
#from chatbot import grounded_gen
from core.prompt_templates import grounded_generation_prompt
from infra.llm_provider import llm

def grounded_gen(nodes, a, b):
    prompt1 = grounded_generation_prompt(nodes, a, b)
    print(prompt1)
    ai_ans = llm(prompt = prompt1, max_tokens=800, stream=True)
    response = ""
    for t in ai_ans:
        u = t["choices"][0]["text"]
        response = response + u
        print(u, end='', flush=True)
    return {"nodes": nodes, "answer": response, "further_instruct": """Below is a grounded generation by AI that cites source from the web. \
You may now write the final answer to the user incorporating the suggested answer, while also adding your own thoughts as you deem fit. \
Be sure to respect any facts cited and do not override those. *Important*: Please copy the inline citation tags verbatim if you reuse those texts."""}
```

Here we add a function that can execute our custom query pipeline.

Finally, we wire up the function calling in the main system:

```py
def web_search_synthesis(search_query: str, topic_question: str) -> str:
    """Synthesize an answer, with inline citations, through searching the web 
    and then passing it through a RAG (Retrieval Augmented Generation) pipeline 
    with a LLM AI.

    The topic_question will be used to select the most relevant text snippets
    in the web search results to feed to the LLM. That LLM will generate an answer
    to the topic_question using the info from the text snippets.

    search_query: Query for the web search engine.
    topic_question: Question to generate answer on.

    Return: Text answer with inline citations.
    """
    do_web_search_synthesis(search_query, topic_question)

def do_web_search_synthesis(a, b):
    print("do_web_search_synthesis")
    print(a, b)
    web_search_results = web_search(a)
    print(web_search_results)
    mydocs = gen_docs(web_search_results)
    print(len(mydocs))
    chroma_client = chromadb.EphemeralClient()
    try:
        chroma_client.delete_collection("temp")
    except BaseException as e:
        print(e)
    embed_model = HuggingFaceEmbedding(model_name="intfloat/e5-small-v2", embed_batch_size=200)
    topic_q = b
    if b is None or b == "":
        topic_q = "Give a relevant summary of the search query: " + a
    mynodes = query_crawl(mydocs, topic_q, 5, embed_model, chroma_client)
    print(mynodes)
    #nodes[0].metadata, nodes[0].text
    return grounded_gen(mynodes, a, b)
```

Note that we use a locally run huggingface embedding model.
