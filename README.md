
## CSV Sample 
| **name**                                                                                        | **caption**                                                            | **link**                                                                                                                                                                   | **icon**                                                                      | **hasBenchmark** |
|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|------------------|
| gpt-4o                                                                                          | Chat completion                                                        | https://ai.azure.com/explore/models/gpt-4o/version/2024-11-20/registry/azure-openai/latest                                                                                | https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg         | true             |
| Phi-4                                                                                           | Chat completion                                                        | https://ai.azure.com/explore/models/Phi-4/version/1/registry/azureml/latest                                                                                               | https://ai.azure.com/modelcache/provider-cache/phi-dark-aistudio.svg          | false            |
| o1-preview                                                                                      | Chat completion                                                        | https://ai.azure.com/explore/models/o1-preview/version/1/registry/azure-openai/latest                                                                                     | https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg         | true             |
| o1-mini                                                                                         | Chat completion                                                        | https://ai.azure.com/explore/models/o1-mini/version/1/registry/azure-openai/latest                                                                                        | https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg         | true             |


## JSON Sample ( models )
``` json
[
  {
    "name": "gpt-4o",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/gpt-4o/version/2024-11-20/registry/azure-openai/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg",
    "hasBenchmark": true
  },
  {
    "name": "Phi-4",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/Phi-4/version/1/registry/azureml/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/phi-dark-aistudio.svg",
    "hasBenchmark": false
  },
  {
    "name": "o1-preview",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/o1-preview/version/1/registry/azure-openai/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg",
    "hasBenchmark": true
  },
  {
    "name": "o1-mini",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/o1-mini/version/1/registry/azure-openai/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg",
    "hasBenchmark": true
  }
]
```
## JSON Sample ( full )
``` json
[{
    "id": 6,
    "name": "gpt-4o-mini",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/gpt-4o-mini/version/2024-07-18/registry/azure-openai/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/aoai-dark-aistudio.svg",
    "hasBenchmark": true,
    "details": "GPT-4o mini enables a broad range of tasks with its low cost and latency, such as applications that chain or parallelize multiple model calls (e.g., calling multiple APIs), pass a large volume of context to the model (e.g., full code base or conversation history), or interact with customers through fast, real-time text responses (e.g., customer support chatbots).\nToday, GPT-4o mini supports text and vision in the API, with support for text, image, video and audio inputs and outputs coming in the future. The model has a context window of 128K tokens and knowledge up to October 2023. Thanks to the improved tokenizer shared with GPT-4o, handling non-English text is now even more cost effective.\nGPT-4o mini surpasses GPT-3.5 Turbo and other small models on academic benchmarks across both textual intelligence and multimodal reasoning, and supports the same range of languages as GPT-4o. It also demonstrates strong performance in function calling, which can enable developers to build applications that fetch data or take actions with external systems, and improved long-context performance compared to GPT-3.5 Turbo.\nResources\nOpenAI announcement"
  },
  {
    "id": 7,
    "name": "Llama-3.3-70B-Instruct",
    "caption": "Chat completion",
    "link": "https://ai.azure.com/explore/models/Llama-3.3-70B-Instruct/version/3/registry/azureml-meta/latest?",
    "icon": "https://ai.azure.com/modelcache/provider-cache/meta-dark-aistudio.svg",
    "hasBenchmark": true,
    "details": "The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.\nBuilt with Llama\nModel Architecture: Llama 3.3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.\nTraining Data\nParams\nInput modalities\nOutput modalities\nContext length\nGQA\nToken count\nKnowledge cutoff\nLlama 3.3 (text only)A new mix of publicly available online data.70BMultilingual TextMultilingual Text and code128kYes15T+*December 2023\n*Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability."
  }]
```
