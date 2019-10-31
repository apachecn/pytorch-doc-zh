# Code plugin for GitBook

Code blocks are cool but can be cooler. This plugin adds lines numbers for multi-line blocks and a copy button to easily copy the content of your block.

## Cool, can I see it working?

The next image shows a single line code block:

![single line](https://github.com/davidmogar/gitbook-plugin-code/blob/resources/images/single.png?raw=true)

When displaying code with multiple lines, line numbers will be added:

![multi line](https://github.com/davidmogar/gitbook-plugin-code/blob/resources/images/multi.png?raw=true)

## How can I use this plugin?

You only have to edit your book.json and modify it adding something like this:

```json
"plugins" : [ "code" ],
```

This will set up everything for you. If you want to get rid of the copy buttons use add this section too:

```json
"pluginsConfig": {
  "code": {
    "copyButtons": false
  }
}
```
