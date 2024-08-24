# Productionizing LLM app with Burr

Turning a prototype LLM GenAI app into production involves many additional considerations. In this tutorial, we will look at a lightweight but useful option: Burr.

https://github.com/dagworks-inc/burr

Burr is a library that at its core, provide abstractions for state machines in general but with additional toolings for going to productions. Two basics (but not all) features are state persistence and monitoring/tracing.

The official docs is pretty good, so let's follow it.

## The classic example - Chatbot

> Reference: https://github.com/DAGWorks-Inc/burr/blob/main/examples/simple-chatbot-intro/notebook.ipynb and https://burr.dagworks.io/getting_started/simple-example/

First install Burr: `pip3 install "burr[start]"`.

```py
import uuid
from typing import Tuple

import openai  # replace with your favorite LLM client library

from burr.core import action, State, ApplicationBuilder, when, persistence

# Note: you can replace this with your favorite LLM Client
client = openai.Client()

@action(reads=[], writes=["prompt", "chat_history"])
def human_input(state: State, prompt: str) -> State:
    chat_item = {
        "content": prompt,
        "role": "user"
    }
    return state.update(prompt=prompt).append(chat_history=chat_item)


@action(reads=["chat_history"], writes=["response", "chat_history"])
def ai_response(state: State) -> State:
    content = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=state["chat_history"],
    ).choices[0].message.content
    chat_item = {
        "content": content,
        "role": "assistant"
    }
    return state.update(response=content).append(chat_history=chat_item)
```

Here we already see much of the core concepts of `burr` at a glance. Let's recall some computer science. A state machine contains states connected by edges. Depending on condition, a state may go through different edges to transition to new state. In the simplest case, an interactive chat between user and LLM is a state machine with two states:

1. In the first state, we wait for human input.
2. Then in the second state, we wait for the LLM reply, after which we go back to state 1 above.

Next, recall that there are variants of state machines depending on the hierarchy of complexity/expressiveness, but in general the philosophy is to have some extra "things" outside of the state that will interact with the state machine, both in terms of influencing the state transition, as well as in terms of the side effect a state transition may have on those "things".

In `burr`, the basic model is that the state machine's states are called `action`. The external thing is, perhaps a bit confusingly, called a `state`. A state here is a dict. In the example here, the schema of the state is:

```py
{
    "prompt": "most recent user prompt",
    "response": "most recent LLM reply",
    "chat_history": [] # Message list in OpenAI format
}
```

State transitions (of the main state machine), especially their side effects, are enacted through executing the `action` itself. As we see above, an `action` is a function taking the current `state` and optional extra arguments, and return the updated `state`. For convinience, functions to modify the state without having to mutate manually is provided by the library: for instance `state.update(key=new_value)` return a state where the `key`'s value is changed to `new_value` but otherwise other keys are unaffected. (In fact `burr` *requires* you to use this kind of functional pattern on the `state` object) Actions would also need to be annotated with hints about what keys in the `state` will it read or write.

> Note: Experienced software developer may begin to feel that this is just the redux pattern in ReactJS. Yes, and as you may have guessed, the action may have side effects other than updating the state. However, only the state object will be explicitly managed by burr, and (?) usual caveat on making your action mostly idempotent? (or at least, idempotent in the sense of not causing damage if rerun with the exact same input)

Perhaps this will be more clear once we also show the main program's code:

```py
app = (
    ApplicationBuilder()
    .with_actions(human_input, ai_response)
    .with_transitions(
        ("human_input", "ai_response"),
        ("ai_response", "human_input")
    ).with_state(chat_history=[])
    .with_entrypoint("human_input")
    .build()
)
```

After specifying the state machine's state (i.e. action) and edge (transition), and then the initial state (main state machine) and state (the side effect one), the app is built.

The transition means that the two possible way to change the main state, is either:

- If you're currently at `human_input`, after executing the side effect, you'd move into `ai_response`.
- If you're currently at `ai_response`, after executing the side effect, you'd move into `human_input`.

In this example, at both state, there is only one possible transition.

Finally we can run it:

```py
*_, state = app.run(halt_after=["ai_response"], inputs={"prompt": "Who was Aaron Burr?"})
print("answer:", app.state["response"])
print(len(state["chat_history"]), "items in chat")
```

This will run only one "loop" and then exit as we designated `ai_response` as a state we will halt after it is executed.

> TODO: Note about the infamous "off by one" problem?

You may then follow the official quickstart and install additional toolings to use the UI, visualize the graph, etc.

## Conditional transitions

> Reference: https://github.com/DAGWorks-Inc/burr/blob/main/examples/simple-chatbot-intro/notebook.ipynb

So what if we want more than one possible transition? Here's an example, extending the one above:

In this extended example, we add a simulated safety check that will check the user's prompt, and if deemed unsafe, will transition instead to the `unsafe_response` action that generate a canned response. First implement the two actions:

```py
@action(reads=["prompt"], writes=["safe"])
def safety_check(state: State) -> Tuple[dict, State]:
    safe = "unsafe" not in state["prompt"]
    return {"safe": safe}, state.update(safe=safe)


@action(reads=[], writes=["response", "chat_history"])
def unsafe_response(state: State) -> Tuple[dict, State]:
    content = "I'm sorry, my overlords have forbidden me to respond."
    new_state = (
        state
        .update(response=content)
        .append(
            chat_history={"content": content, "role": "assistant"})
    )
    return {"response": content}, new_state
```

Notice that in `burr`, there are multiple way to define an action, and the actual conceptual model is more complex - it allows returning output at each action aside from the state update too. A third way is to use class based action.

```py
safe_app = (
    ApplicationBuilder().with_actions(
        human_input=human_input,
        ai_response=ai_response,
        safety_check=safety_check,
        unsafe_response=unsafe_response
    ).with_transitions(
        ("human_input", "safety_check"),
        ("safety_check", "unsafe_response", when(safe=False)),
        ("safety_check", "ai_response", when(safe=True)),
        (["unsafe_response", "ai_response"], "human_input"),
    ).with_state(chat_history=[])
    .with_entrypoint("human_input")
    .build()
)
safe_app.visualize(output_file_path="digraph_safe", include_conditions=True)
```

The main thing here is that a condition transition is created with the tuple `(cur_action, next_action, condition)`. `when()` is a function to create a condition on detecting whether a key in the state has a specific value. With this, the state machine is "routed" - executing an action result in side effect change to the state, which then affect which transition is taken, etc.

## Tracking and persistence

And now onto the actual thing with getting to production. First thing is that a run of a `burr` app can be tracked with full history and reviewed. This is done simply by adding a line when building your app:

```py
app = ... .with_tracker(
        "local", project="demo_getting_started"
    ).build()
```

Then, run `burr` in the terminal, and open the web UI.

As to persistence, because the model explicitly control and manage the transitions and state updates, it is possible to persist an app and restore it later. In the context of the basic example above, this will let us save an ongoing conversation to a database and restore it later.

Like the official notebook says, To add a persistor, you have to tell it to load from a state (`.initialize(...)`) on the builder, and tell it to save to a state (`.with_state_persister`).

```py
# we're going to be creating this multiple times to demonstrate so let's stick it in a function
def create_persistable_app():
    sqllite_persister = persistence.SQLLitePersister(db_path="./sqlite.db", table_name="burr_state")
    sqllite_persister.initialize()
    return (
        ApplicationBuilder().with_actions(
            human_input=human_input,
            ai_response=ai_response,
            safety_check=safety_check,
            unsafe_response=unsafe_response
        ).with_transitions(
            ("human_input", "safety_check"),
            ("safety_check", "unsafe_response", when(safe=False)),
            ("safety_check", "ai_response", when(safe=True)),
            (["unsafe_response", "ai_response"], "human_input"),
        ).initialize_from(
            initializer=sqllite_persister,
            resume_at_next_action=True,
            default_state={"chat_history": []},
            default_entrypoint="human_input"
        ).with_state_persister(sqllite_persister)
        .with_identifiers(app_id=app_id, partition_key=partition_key)
        .with_tracker(
            "local", project="demo_getting_started"
        ).build()
    )
```

Let's explain this with a few things.

**Persistor**

A persistor abstract the low level interaction with the underlying persistence mechanism, which could be various kinds of database, files, etc. Some common types are provided officially, though you can implement your own if you have special need.

**Keying**

We need to have an `app_id` and a `partition_key`. The `app_id` identify the app, because in production we may have many different kinds of app running at the same time. The `partition_key` is used to distinguish different runs of the same app - such as different conversation thread.

**Specifying save and load method**

`.initialize_from()` tell the app how to load from a persistor. `.with_state_persister()` tell the app to use that persistor to save.

Then, we need `.with_identifiers(app_id, partition_key)` to pin down *which* run we'll be saving/loading.

## Integrating with Web Server

The interesting thing is, you don't need additional concepts! It is just a library, how you use it is up to you. That being said, https://github.com/DAGWorks-Inc/burr/blob/main/examples/web-server/README.md provide a guide of the programming pattern that you may consider for a manual integration.
