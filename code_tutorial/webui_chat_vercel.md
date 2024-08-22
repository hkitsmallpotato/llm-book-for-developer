# Building Web UI for AI Chatbot with Vercel AI SDK and Chatkit-UI

In this example, we'll show how to use Vercel AI SDK to quickly interact with an OpenAI API compatible endpoint in the frontend world.

## Background

Vercel AI SDK is a collections of library that makes it easy to get started developing frontend for LLM based app. In alignment with its frontend focus, it uses an isomorphic tech stack (i.e. JS on both frontend (ReactJS running on browser JS runtime) and backend (the NodeJS runtime)). This presents a problem though as python is the dominant language for AI.

Two of the libraries we will use are:
- **Vercel AI SDK Core**: Handle calling the OpenAI API
- **Vercel AI SDK UI**: Provide convinience by having the `useChat` React hook so that integrating an OpenAI API endpoint to the frontend is more intuitive.

Notice that we will use NextJS which is both the frontend and the BFF (Backend for frontend) together. Throughout, the pattern will be:

> Frontend -> BFF (Contain API key) -> OpenAI API

(TODO: list documentation links for quick reference)

## Project Setup

1. Scaffold a NextJS project with the latest version.
2. Install dependencies:

```
npm install ai @ai-sdk/openai zod @chatscope/chat-ui-kit-react @chatscope/chat-ui-kit-styles
```

Note that:
- `ai` is the vercel AI SDK
- For each AI Model provider, we need to install and import their specific provider package. For Open AI, that'd be `@ai-sdk/openai`.

## Basic integration

In the first pass, we'll build a basic webpage with a button you can click to submit a LLM text completion request. We will need to do three things:

As we may need to connect to third party LLM API provider (which are OpenAI API compatible), we need to change the base URL. We also, of course, need to supply the API key.

1. **Provider Setup**: Create a file `./app/registry.ts`:

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { experimental_createProviderRegistry as createProviderRegistry } from "ai";

export const registry = createProviderRegistry({
  // register provider with prefix and custom setup:
  openai: createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: "https://example.com/v1", // Replace with your provider endpoint here
  }),
});

```

`createOpenAI` is used to instantiate a provider, which we then configured/customized. We then register it, and now the registry knows that "openai" refers to that instance.

2. **Create a server side endpoint** in NextJS that calls the LLM: Create the file `./app/actions.ts`:

```ts
"use server";

import { registry } from "./registry";

import { generateText } from "ai";
//import { openai } from "@ai-sdk/openai";

export async function getAnswer(question: string) {
  const { text, finishReason, usage } = await generateText({
    //model: openai("gpt-4-turbo")
    model: registry.languageModel("openai:Meta-Llama-3.1-8B-Instruct"),
    messages: [
      {
        role: "system",
        content:
          "You are a helpful AI assistant for answering simple question. You give crisp, concise answer.",
      },
      {
        role: "user",
        content: question,
      },
    ],
  });

  return { text, finishReason, usage };
}
```

Notice a few things:
- We must add `"use server";` to let NextJS know this is a server side component.
- Instead of instantiating a default OpenAI provider `openai("gpt-4-turbo")`, we retrieve the custom provider we prepared earlier via `registry.languageModel("openai:Meta-Llama-3.1-8B-Instruct")`. The string `"openai:Meta-Llama-3.1-8B-Instruct"` means to ask the registry to retrieve the provider registered under the name of `openai`, and set it to call the LLM model "Meta-Llama-3.1-8B-Instruct" when making OpenAI API calls.
- This is only a basic example, so we used the simpliest `generateText` method. Refer to the docs for more sophisticated use case such as Streaming or tool use.
- The method supports the following for specifying prompt, in recognition of the OpenAI API's `/completions` and `/chat/completions` endpoints:
  - system: Specify a system message
  - prompt: Directly give a prompt
  - messages: List of messages in OpenAI API format, for chat completion

3. **Create the frontend component**: Modify `./app/page.tsx` to have the following:

```tsx
"use client";

import { useState } from "react";
import { getAnswer } from "./actions";

export default function Home() {
  const [generation, setGeneration] = useState<string>("");

  return (
    <div>
      <button
        onClick={async () => {
          const { text } = await getAnswer("Why is the sky blue?");
          setGeneration(text);
        }}
      >
        Answer
      </button>
      <div>Hello</div>
      <div>{generation}</div>
    </div>
  );
}
```

Notice that:
- It is a client component (compare to 2 above)
- Recall that in NextJS with app routing, the file named `page.tsx` will be implementation of frontend for a route given by the folder path, this is by convention.

Remember to set the LLM API key through the enviornment variable `OPENAI_API_KEY` and then run the project. Now you may click the button and see the response on screen.

## Chat UI integration

Next, we will add a Chat UI. We will use the Chatkit UI project to provide pre-designed UI components.

1. **Create a custom API endpoint:** this endpoint will need to fulfill some request/response schema for it to be usable by the `useChat` ReactJS hook. Create a file `./app/api/chat/route.ts`, which will provide the endpoint at a URL that is pointed to by default by the `useChat` hook:

```ts
import { registry } from "../../registry";
import { streamText } from "ai";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = await streamText({
    model: registry.languageModel("openai:Meta-Llama-3.1-8B-Instruct"),
    messages,
  });

  return result.toDataStreamResponse();
}
```

This is similar to the first example, except that we used `streamText` in the SDK Core library. We also made it accessed through HTTP POST method.

2. **Create the frontend.** This is more complex, so let's break it down. In the file `./app/chat/page.tsx` (For the route `/chat`):

```tsx
"use client";

import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";

import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  Avatar,
  TypingIndicator,
} from "@chatscope/chat-ui-kit-react";

import React, { useState } from "react";

import { useChat } from "ai/react";
```

We made it a client component, imported the ChatUI kit CSS styles and the core UI components. We also imported the `useChat` hook.

```tsx
function MyMessage({ model, keyi }) {
  var mapped_model = {
    direction: model["role"] == "assistant" ? "incoming" : "outgoing",
    message: model["content"],
  };
  return (
    <Message key={keyi} model={mapped_model}>
      {mapped_model["direction"] === "incoming" ? (
        <Avatar
          name="Joe"
          src="https://chatscope.io/storybook/react/assets/joe-v8Vy3KOS.svg"
        />
      ) : (
        false
      )}
    </Message>
  );
}
```

As ChatUI Kit have a different data model then the OpenAI standard, we perform custom conversion here. The `keyi` argument is simply an integer coming from array index, for React to render list and deal with the rerender performance problem/to help the reconciliation algorithm. Let's compare the two data models:

OpenAI:
```json
{
    "role": "user",
    "content": "Text of the message."
}
```

(`role` can be `"system"`, `"user"`, or `"assistant"` at minimal. Some implementation may supports more such as for tool calling)

ChatUI Kit:
```json
{
    "direction": "incoming",
    "message": "Text of the message."
}
```

(`direction` is either `"incoming"` or `"outgoing"`.)

```tsx
export default function Chat() {
  const { messages, append, isLoading } = useChat();

  const sendMessage = (...[, , innerText]) => {
    console.log(innerText);
    append({
      role: "user",
      content: input,
    });
  };
```

We call the `useChat` hook and extract the parts we need:
- `messages` contain the full conversational history so far and is reactive, so we can directly use this to display the messages.
- `append` is used to trigger calling the default BFF endpoint to request a text/chat completion (in this case, chat completion) from the underlying LLM API. It is so named because we effectively "append" an entry to the OpenAI formatted message list.
- `isLoading` can be used to disable input box and display a loading state as appropiate when request is submitted and we're waiting for response.

```tsx
  return (
    <div style={{ position: "relative", height: "500px" }}>
      <MainContainer>
        <ChatContainer>
          <MessageList
            typingIndicator={
              isLoading ? <TypingIndicator content="Joe is typing" /> : false
            }
          >
            {messages.map((message, index) => (
              <MyMessage model={message} keyi={index} />
            ))}
          </MessageList>
          <MessageInput
            //value={input}
            disabled={isLoading}
            placeholder="Type message here"
            sendOnReturnDisabled={true}
            //onChange={updateMessage}
            onSend={sendMessage}
          />
        </ChatContainer>
      </MainContainer>
    </div>
  );
}
```

This is the main UI:
- Notice how `isLoading` controls both the typing indicator and the disable state of the input box
- We map over the state `messages` and use our custom UI component `MyMessage` defined earlier.

What is interesting here is that we get fairly low level to make the integration of ChatUI Kit with Vercel AI SDK UI. If using a vanilla HTML form and input, the hook have the default lifecycle methods `handleInputChange` and `handleSubmit` that can be used directly, then defining custom function is not required (TODO: is the binding of `input` state from the hook into the `<input/>`'s `value` attribute a bidirectional data binding?) Alternatively, `input` plus `setInput` from the hook can also be used directly to control the value. Ultimately, they seems to stem from the need to get the input value when the UI's event handler doesn't have that value somewhere in the function argument.



