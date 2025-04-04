# Code Citations

## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsque
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1)
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.
```


## License: unknown
https://github.com/UkiDLucas/Aiko/blob/962ee4ed6d029b6cdaedd666dc13f95f13f2b69d/GPT/python/hugging_face_transformers_usage.py

```
mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-
```

