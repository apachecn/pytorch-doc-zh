

# Serialization semantics

## Best practices

### Recommended approach for saving a model

There are two main approaches for serializing and restoring a model.

The first (recommended) saves and loads only the model parameters:

```py
torch.save(the_model.state_dict(), PATH)

```

Then later:

```py
the_model = TheModelClass(*args, **kwargs)
the_model.load_state_dict(torch.load(PATH))

```

The second saves and loads the entire model:

```py
torch.save(the_model, PATH)

```

Then later:

```py
the_model = torch.load(PATH)

```

However in this case, the serialized data is bound to the specific classes and the exact directory structure used, so it can break in various ways when used in other projects, or after some serious refactors.

