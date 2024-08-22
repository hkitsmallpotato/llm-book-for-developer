# (Appendix) LLM Open Weight Model Recommendation

There are numerous models out there. For the purpose of GenAI app development however, we might be able to narrow down the choice.

For beginners, using the mainline models by large organization is recommended as the more well known models may have better ecosystem supports in general. A second consideration is size. As of 2024, compute constraint is still very real, so you'll have to check your computer's spec - with special focus on amount of RAM (and VRAM if using GPU), and memory bandwidth.

To make things short, below are my general conclusion (Note: although not specified, please choose the instruct version of the models unless you specifically want a base model):

**For smoke testing/development and when the LLM's output quality doesn't matter**:

The scenario is you're mainly testing the technical integrations such as whether the API call works etc. You may use SLM (Small Language Model) instead at the sub-1B param range, which largely eliminates any compute concern (they should be able to run on any modern hardware quickly).

Models:
- H2O's Danube3 500m
- Huggingface's SmolLM 360m (v0.2)
- Qwen2 0.5B

**For General use, edge deployment or CPU only**

You want to have reasonable capability plus speed, but are unable to procure a GPU computer. In this case models at the 2-4B range is an options. The best of breed here is undoubtedly Google's **Gemma 2 2b**, which is punching above its weight.

Other options include:
- The recent Minitron series by Nvidia at 4B
- Microsoft Phi 3.5 mini (3.8B)
- Stability AI (+ Huggingface) StableLM series: stabilityai/stablelm-2-zephyr-1_6b and stabilityai/stablelm-zephyr-3b etc.

You may generally expect to get reasonable general conversation performance, however it is likely to struggle for advanced use case such as agents.

**For development and with access to at least 6GB VRAM GPU/3xxx series from Nvidia**:

Assume you have GPU with enough VRAM, you may begin to run what is considered mainline models (not the large one minds you, but still).

Main options:
- llama 3.1 8b from meta
- Gemma 2 9b from Google
- (Older) Mistral v0.3 7b
- (Older) WizardLM 2 7b (technically a fine tune)

At this level, you may also begin to consider using community fine-tune models (but beware that if you intend to productionize it, not all cloud vendor may offer all fine-tunes as a shared inference API service, this will have economic implications). Especially for fine-tunes trained for function calling, agents...

Example:
- Hermes series by NousResearch


## Note

- For productionizing, please remember to check the License first (Open-weight encompass a wide range of license in terms of how open it actually is and whether there are any legal restrictions). If unsure, consult your lawyer.
- If you want to work with the 7-9b range models but only have CPU, you may still try - expect token/second in the range of 3 - 15 (in case of Apple unified memory) roughly speaking - it depends on the details of your computer.
- Non-nvidia GPU is sadly a wild west world - it might work, but it's somewhat of a gamble.
