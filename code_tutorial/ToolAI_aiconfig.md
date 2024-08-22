# ToolAI via aiconfig

> TODO: This is just copying from official doc with curation. Will have to implement custom extension before we can actually use it though.

`aiconfig` turns prompts into config data that can both be manipulated programmatically, but also interactively through human-centric tools. Secondly, it provides a lightweight execution engine that is suitable for the ToolAI use case, where a LLM invokation with prompt templating can be thought of as a natural language function. How you wire these functions up though is however up to you: you may treat them like ordinary functions in a computer program and well, program it. Or you may opt for integration into a workflow orchestration engine for instance.

> TODO: find a good use case to demo it. Currently Generating the creative text assets of a game, especially role playing game with world building, is a good choice, but I think I may hold on a bit first.

## Install and Using the IDE/WebUI

To begin, install through `pip3 install --user python-aiconfig`, setup LLM access (OpenAI API key if you have access to the actual OpenAI), then either open WebUI through `aiconfig edit --aiconfig-path=travel.aiconfig.json`, or install the VSCode Extension.

Follow the guide at https://aiconfig.lastmileai.dev/docs/getting-started to use the UI, though you may want to learn the concepts first - see next section.

## Core Concept: Prompt Chain

As we know, a prompt chain is where outputs from one LLM invokation is injected into the prompt template of another LLM invokation, and so on. Taking the view above that one single LLM invokation, if wrapped with a prompt template, can be conceptualized as a natural language function, then the injected values can be thought of as the "function arguments/inputs", and the LLM output is the "function return value". Prompt chaining is then simply a form of complex function composition where the values are wired up and pass around in potentially complex ways. From traditional computer science, one useful way to model this is a workflow/directed acyclic graph/computational graph. Here each node is a function, the lines connecting nodes represent values flowing from output of one node/function, into some function argument of another node.

### Parametrized Chain

For ToolAI, a good model is the Parameter Chain in aiconfig. 

As we discuss above, we will need a **prompt template** - a prompt/text that has template variables that the library then substitute values into. Here are the details:

- Uses `{{handlebars}}` syntax
- Values passed between different prompts/nodes through `{{<prompt_name>.output}}` and `{{<prompt_name>.input}}` respectively. (Hopefully self-explanatory, if not, refer to official docs.)
- A plain variable `{{ var1 }}` would be dangling parameter that user/you will provide the value of. There are multiple ways to do this:
  - At runtime, per function invokation.
  - Default value provided as part of the config file.
  - A default value set at runtime through code.
- Special note: `aiconfig` seems to have several reserved templating variables that are always active: prompt, system_prompt, and the converation history.

### Conversational Chain

Although ToolAI is nice, in practise, chat based LLM models have dominated purely instruct based one. One common way to nevertheless use them as ToolAI is through "pre-programmed dialog/chat" - in an interactive session the human user manually supply the user turn messages one by one, based on the reply by AI. In contrast, in a pre-programmed session, we may simply hardcode the "user" reply on each turn.

> TODO: Example of how this is useful.

`aiconfig` have provisioned for this use case with a twist: by default, if multiple prompt node uses the same model, then they'd be considered part of one **Conversational Chain**. In such case chain invokation is *stateful* - `aiconfig` will automatically remember the conversation history, and for each subsequent calls of individual nodes, the prompt will instead be inserted as a new message in the chat history before being sent to the LLM.

To disable this, you may do so on a per-node basis by (See official doc and also see sections below first):

```py
config.set_metadata("remember_chat_context", False, "prompt2")
```

## Basic Usage

To run a chain in `aiconfig`, there are two basic steps:

### Load a chain and set values

```py
from aiconfig import AIConfigRuntime

# Load the aiconfig.
config = AIConfigRuntime.load('sql.aiconfig.json')

# Set a global parameter
config.set_parameter("sql_language", "mysql")

# Set a prompt-specific parameter
config.set_parameter(
    "output_data",
    "user_name, user_email, trial. output granularity is the trial_id.",
    "write_sql" #prompt_name
)

config.save()
```

Above is an example from official doc. We load the file, then use `set_parameter(param_name, value, prompt_name)` to set it. The distinction between global parameter and parameter scoped to a single prompt is present in the config file.


### Run the chain

At its heart, `aiconfig` is similar to many DAG workflow execution engine. So to execute a prompt chain:

```py
from aiconfig import AIConfigRuntime, InferenceOptions

# ... Load config and set params

inference_options = InferenceOptions(stream=True) # Defines a console streaming callback

result = await config.run("prompt_name", params, options=inference_options)
```

Here `params` is a python dict containing values of any parameter you haven't set in the config beforehand. `aiconfig` will do the usual Computer Science stuff of computing the transitive closure, then resolving all parameters, and then execute the prompts according to dependency order.

Notice the following:

- By default, output of a node is cached. If you want a full run again, add `run_with_dependencies=True` as an additional arguments to `config.run()`.
- Because the nodes are a DAG, there is not necessarily a single "terminal node" - it is entirely possible to run the chain only up to some intermediate node. Similarly, because a DAG can be a forest (i.e. have more than one connected component), it is also perfectly possible to have an isolated node with no connections with other node and we only execute that single node.

... and, that's it! Congratulations! You are now free to integrate this into your app any way you want to. (This flexibility and light-weightedness is part of its strength)


## More

### Notes and Extension

> TODO

- Metadata is generic - you can add anything you want there. To set a metadata, use `set_metadata()`. (Note however that some metadata name is reserved and has special meaning)
- Although by default `aiconfig` uses the `{{ handlebar }}` templating syntax, it is possible to define your own syntax and override it.

## Practical

The power of `aiconfig` lies in its simplicity. At the same time, this simplicity (and heavy dependence on knowing about DAG) may makes it difficult to see how it can be used for various real world use case.

> Key insights: As single node prompt and a prompt chain is treated uniformly, as part of a larger DAG, one may use prompt chain if the flow is known before-hand, or invoke single node in a dynamic manner for fine-grained control.
