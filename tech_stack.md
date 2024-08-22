# Tech Stack for LLM GenAI Applications

- As this is an emerging technology, software libraries and especially frameworks related to it are still in the early stage, in rapid flux, and often not mature yet
- Lessons from history of web framework? (Backend: MVC, incremental change when RESTful API become mainstream while co-evolving with the rise of Single Page Application in frontend; Frontend: early days like backbone.js and MVVM, debates on uni-vs-bidirectional data binding, state management, eventual dominance of React, but less settled for Server-Side Rendered (SSR) or hybrid frontend (Yes there is Next.JS, but); Infra: Container orchestration war)
  - Changes take time; but once the preparation is ready phase shift can be fast
  - It can be difficult to see what's ultimately considered the right pattern in early days when we're still in exploration and co-evovlution stages; early players that's dominating is not necessarily what will eventually become industry standard?
  - "Incidental Factors" may also play a role; for example the historical context of Javascript and lack of quality-of-life features (such as issue with typing and module), as well as the culture that colase around it, led to tooling nightmare that took a relatively long time to resolve
  - Convergence effect of having a standard interface at key middle point in the stack (?): lower layers eventually reach feature parity/commodity status
- Issues specific to LLM GenAI app:
  - Cultural divide: People with backgrounds in AI are not necessarily experienced software engineer, and vice versa
  - Multiple domain: it remains to be seen whether the monoculture paradigm in WebDev is applicable to this nascent field, as there are currently four over-lapping, but distinctive "style" of LLM GenAI Apps (ToolAI, Chat and RAG, Agents, Copilot). This may makes it harder to develop a complete framework (though it is questionable whether a complete framework is desirable in the first place per the monoculture remarks above), makes the scene more suceptible to noise and harder to gain clarity on the full picture, increases risk of ecosystem fragmentation and tribal fanfare, etc.

## Survey of options

- Distinguish between libraries suitable for exploration/tinkering/proof of concepts/prototyping/production

(TODO) Have some comparison table?

### UI/Frontend

- Base is standard React or NextJS
- Consider using TanStack Query for calling ToolAI endpoint
- Chat interface: https://github.com/chatscope/chat-ui-kit-react
- Integration and UI quality-of-life: Vercel AI SDK Core and UI (provides React hook) https://sdk.vercel.ai/

https://spin.atomicobject.com/basics-tanstack-query/
https://tanstack.com/query/latest/docs/framework/react/overview
https://dev.to/john_dunn_ec1dda9d69d5743/getting-started-with-tanstack-query-5b58

### ToolAI

- AIConfig by lastmile ai https://aiconfig.lastmileai.dev/
  - More limited scope and utility approach than LangChain
  - Integrated prompt editor UI, but also ultimately code/data based and programmable (Good!)
  - Supports for alternative model provider might need work though (Very common weakness of many libraries and frameworks in this space is that they assume OpenAI API, which is not the worst due to OpenAI API compatible provider, the worst is when the config to switch to a different Base URL is not present, or when the model name is either not parametrized, hidden deep in library internal, or spread throughout the library)

### Structured/Constrained/Guided Output

- Challenging to compare options
- Too many dimensions, and sometimes inter-locking dimensions
- Lots of room for cherry-picking, so please do your own eval based on your actual need
- Just to name a few off the top of my head (not necessarily exhaustive):
  - Backend: support advanced features like compatible with beam search, streaming + partial parse?
  - Backend: Performance optimization on the FSM sync algorithm?
  - Backend: Although Context-Free-Grammar (CFG) is in a sense the fully general solution, does the library support optimizations for more specific subclass? (eg JSONSchema)
  - Backend/Philosophy: Compatible with both self-hosted and OpenAI? How to deal with discrepency/difference in features? (May be a temporary concern due to the convergence effect above)
  - Philosophy: Should scope be limited to just generating structured object, or more general grammar?
  - Philosophy: Should we advocate for a corrective approach (let LLM generate freely but ask for self-repair if anything goes wrong), the algorithmic one (Use FSM synchronization to gaurantee technical adherence), or be a pragmatic and stay neutral?
  - Philosophy/Backend (Due to performance consideration): Should we focus on single generation, or consider the whole text completion in a holistic manner and design a DSL (Domain Specific Language) for interleaved prompting (maybe with tool use interception) and generation?
  - Misc: Is the "Abuse function calling" trick okay to use?
  - Frontend: How to integrate the DSL into program? DSL-in-code with custom compiler or similar intervention, or separate file, or fully code native? (eg JSX in React, lmql embedded in python code comment, guidance, sglang...)
- "Philosophy" because it involves value judgement that may not have an obvious right answer (or even objectively have one - relativism and pluralism is a thing afterall)

**Candidates**: lmql, guidance, sglang, autogen, lm-format-enforcer, instructor, outline...

(Also a new one recently?)

### Agent

**Candidates**: autogen, crewAI, langgraph, langroid...

### Productionizing

**Web Integration**

- langserve (Autogenerate a suite of endpoints with playground)
- burr https://github.com/dagworks-inc/burr
 - Not directly about this (it is mainly about managing state machine for a DAG flow with supports for persistence etc), but may turn out to be surprisingly suitable esp. for state heavy, interactive LLM GenAI App such as Chat.

**Prompt Management**

?

**Monitoring/Tracing**

- langfuse

**Misc**

- litellm

