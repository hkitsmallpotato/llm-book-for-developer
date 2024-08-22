# Running Local LLM Programmatically, and Basics of OpenAI API

There are various ways to interface with a local LLM in code, and some of them involves the OpenAI API, which have become a de facto standard interface. Therefore, we may also do some introduction on that. However, for the main teaching materials on concepts, please refer back to the main book.

## Option 1: Run an OpenAI API compatible server, and then make HTTP call

To do so, we first need to run the server. I recommend either `ollama` or vanilla `llama.cpp` for beginners.

### Starting the server

**Via ollama**:

Simply start the ollama service and the endpoint will be available at `http://localhost:11434/api`.

Alternatively, you can also run `ollama serve` directly in console, which will run the server process in foreground mode.

**Via llama.cpp**:

Run `.\llama-server.exe -m <path to GGUF model file>` (on window, linux is similar), endpoint at `http://localhost:8080/v1`. (Notice that there are also `llama.cpp` specific endpoints, outside of OpenAI API standard, that runs on `http://localhost:8080/`. They used to provide important features not supported by OpenAI API, especially around structured/constrained output. With OpenAI's recent announcement of adding support, we may see this converge.)

Notice that similar to running `llama.cpp` command line program, there are additional options you can supply. Refer back to the running OSS LLM guide (TODO) for details.

**Quick test**:

On Window, you may use `curl.exe` in powershell (instead of `curl`, which points to Window native `Invoke-WebRequest` even though it is arguably more feature rich). You may need to adjust the commands. Below is an example:

```
curl.exe -X POST --url http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "na", "messages":[{"role":"system", "content":"you are a witty AI assistant."}, {"role":"user", "content":"Hello, how are you?"}], "max_tokens": 250, "stop":["<|im_end|>"]}'.Replace('"', '""')
```

Note:
- In window, handling quote escape requires careful treatment. Here we used a hack.
- `llama.cpp` will typically attempt to infer prompt template from the GGUF file, but it may fail for less popular models, and in those case it will fall back to default of **chatml** format. This is why we need the extra stop sequence `"stop":["<|im_end|>"]` above. You may check this by looking at the llama.cpp server log.
- Model name doesn't matter, and no access key is needed.

You can also override the prompt template by supplying command line argument `--chat-template <short-name>` when launching server. Only a preset families are accepted, see https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template (You may also add custom template or use `--in-prefix` and `--in-suffix` as a workaround)

### Making the call in python

Install the `requests` library: `pip3 install requests`

Then run:

```py
import requests
import json
url = "http://localhost:8080/v1/chat/completions"
payload = json.dumps({
  "model": "na",
  "messages": [{"role":"system", "content":"You are an AI assistant."}, {"role":"user", "content":"Hi! How are you today?"}],
  "max_tokens": 250,
  "stop": ["<|im_end|>"]
})
headers = {
  'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
```

For simplicity's sake, streaming have been disabled.

Let's examine the result:
```
>>> response
<Response [200]>
>>> response.json()
{'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': "I'm doing great! You too?", 'role': 'assistant'}}], 'created': 1724344537, 'model': 'na', 'object': 'chat.completion', 'usage': {'completion_tokens': 16, 'prompt_tokens': 56, 'total_tokens': 72}, 'id': 'chatcmpl-ba1CdEyyS40Bp3bzmPAlX86jq9cTvRUY'}
>>>
```

We can extract the text via `response.json()["choices"][0]["message"]["content"]`, which is a string (you can print it also)

### Streaming Response

> TODO

For reference, below is a program that will stream the response:

(Source: https://gist.github.com/lemonteaa/9a6a460b8bf5c13a200f6aee7a049c25)

```py
# Credit: https://www.codingthesmartway.com/stream-responses-from-openai-api-with-python/

import requests
import json
import sseclient

url = "http://localhost:8080/v1/chat/completions"

payload = json.dumps({
  "model": "na",
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Sup! How're you doing? ;)"}
  ],
  "repetition_penalty": 1.1,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "max_tokens": 8000,
  "stream": True
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': f"Bearer no-key"
}

my_req = requests.post(url, headers=headers, data=payload, stream=True)
client = sseclient.SSEClient(my_req)

for event in client.events():
    if event.data != '[DONE]':
        data = json.loads(event.data)
        choice = data['choices'][0]
        if choice["finish_reason"]:
            print(f"\n\n----\nFinish reason: {choice["finish_reason"]}")
            print(f"Token stats: Completion: {data["usage"]["completion_tokens"]}, Prompt: {data["usage"]["prompt_tokens"]}, Total: {data["usage"]["total_tokens"]}")
            print("bye", flush=True)
        else:
            print(choice['delta']['content'], end="", flush=True)
```

Remember to adjust parameter as needed.

### Variation: Use OpenAI SDK client

You may also use the official SDK client by OpenAI.

`pip3 install openai`

```py
from openai import OpenAI
client = OpenAI(
  api_key="no-key",
  base_url="http://localhost:8080/v1"
)

result = client.chat.completions.create(messages=[
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Sup! How're you doing? ;)"}
], max_tokens=250, stop=["<|im_end|>"], model="na")
```

It is broadly similar to the raw HTTP requests we used above, but with some difference. For instance, the result is wrapped into its own object:

```
>>> result
ChatCompletion(id='chatcmpl-AfBfHgrwjA9MA4rM0v9rdRD3jvdiEMsn', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="Hey there, I'm here to help!\n", role='assistant', function_call=None, tool_calls=None))], created=1724345194, model='na', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=60, total_tokens=78))
>>> result.choices[0].message.content
"Hey there, I'm here to help!\n"
>>>
```

(So that attribute access is by dot instead of when it is a dict; similarly no need for doing json conversion by ourself)

> TODO elaborate

One important advantage of the SDK however is that streaming response is significantly easier. See https://cookbook.openai.com/examples/how_to_stream_completions for details. One point to note is that the response object will have a similar but different schema:

```py
for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content) # delta, not message
    print("****************")
```

Moreover, be careful of the stream end. (TODO verify?) In both methods, we see that right before the stream termination at the lower network layer, the last message will have empty object at the `choice` level. The reason is because the API want to provide one last message for telemetry purpose such as counting token usage.

## Option 2: Use the python binding via llama-cpp-python

With this method we can access it in the same python interpreter process. This is useful if very tight integration is needed (eg for implementing structured output, special and custom logit processing beyond what is offered by the API, such as custom sampler or even watermarking). It is also arguably useful for avoiding when server process die unexpectedly. On the other hand, the API makes integration with external system easier.

First install the `llama-cpp-python` library. This is dependent on whether you use GPU or not, and may requires complicated compilation. You can opt for downloading pre-compiled wheel on the github release page instead.

