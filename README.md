# Supplementary Code for LLM-based Decision Tree Induction

More detailed README will be added shortly.

**Installation:** See `pyproject.toml`

**Running:** There are two main notebooks:
1. [`./build_tree.ipynb`](./build_tree.ipynb) for learning a tree from scratch.
2. [`./build_tree_over_tabpfn.ipynb`](./build_tree_over_tabpfn.ipynb) for learning LLM-guided tree that refines TabPFNv2 predictions.

The notebooks rely on an OpenAI-like proxy API by default.

To switch to a public API, replace ProxyAPIModel with `smolagents.LiteLLMModel` and initialize it [according to the documentation](https://huggingface.co/docs/smolagents/v1.2.2/en/reference/agents#smolagents.LiteLLMModel). For open-source models, you can use either HuggingFace API or custom deployment as per the [smolagents API reference](https://smolagents.org/docs/agents-guided-tour/).

___
### Human input experiments:
3. [`./lost_feature.ipynb`](./lost_feature.ipynb) for learning a tree with lost feature.

Fairness will be added shortly.