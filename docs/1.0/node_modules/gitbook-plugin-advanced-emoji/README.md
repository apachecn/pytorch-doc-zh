# GitBook Plugin: Advanced Emoji

Transforms emojis like `:white_check_mark:` into real <img src="https://codeclou.github.io/gitbook-plugin-advanced-emoji/doc/white_check_mark_20.png"/> emoji images using [emojify.js](https://github.com/hassankhan/emojify.js).

Please note that this plugin only works for **markdown** files. All other filetypes will just be ignored, and the plugin does nothing.

## Installation

You can install this plugin via npm:

```bash
$ npm install gitbook-plugin-advanced-emoji
```

Be sure to activate the option from the `book.json` file:

```json
{
    "plugins" : ["advanced-emoji"]
}
```

Then run `gitbook install` followed by either `gitbook serve` or `gitbook build`


## Using Ignore Flags

If you want for example occurences of emojis **not replaced** you will need to wrap them in the following comments.

```
<!-- ignore:advanced-emoji:start -->
:white_check_mark:
<!-- ignore:advanced-emoji:end -->
```

You can even set the ignores around a codeblock or more lines.

```
This is a text

<!-- ignore:advanced-emoji:start -->
'''
Check the Code
Code ... :white_check_mark:
'''
<!-- ignore:advanced-emoji:end -->

foo
```

## Versions

  * GitBook will automatically install the right version of the plugin
    * `master` branch is for GitBook v3.x and plugin version is `0.2.x`
    * `gitbook_v2` branch is for GitBook v2.x and plugin version is `0.1.x`


## Building a PDF

You can see the Branch [pdf-test-book](https://github.com/codeclou/gitbook-plugin-advanced-emoji/tree/pdf-test-book) on how to use the plugin when building a PDF.

## License

 * https://github.com/codeclou/gitbook-plugin-advanced-emoji is licensed under MIT License
 * https://github.com/hassankhan/emojify.js is licensed under MIT License

