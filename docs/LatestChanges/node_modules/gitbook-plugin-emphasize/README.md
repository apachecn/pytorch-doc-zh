# Emphasize texts in GitBook

Emphasize and highlight specific part of your content

### How to use it?

Configure the plugin in your `book.json`:

```js
{
    "plugins": ["emphasize"]
}
```

Then in your markdown/asciidoc content, highlight some text using:

```md
This text is {% em %}highlighted !{% endem %}

This text is {% em %}highlighted with **markdown**!{% endem %}

This text is {% em type="green" %}highlighted in green!{% endem %}

This text is {% em type="red" %}highlighted in red!{% endem %}

This text is {% em color="#ff0000" %}highlighted with a custom color!{% endem %}
```
