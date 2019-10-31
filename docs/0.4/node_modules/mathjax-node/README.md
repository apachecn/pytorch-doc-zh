# mathjax-node [![Build Status](https://travis-ci.org/mathjax/MathJax-node.svg?branch=develop)](https://travis-ci.org/mathjax/MathJax-node)

This repository contains files that provide APIs to call [MathJax](https://github.com/mathjax/mathjax) from 
node.js programs.  There is an API for converting individual math 
expressions (in any of MathJax's input formats) into SVG images or MathML 
code, and there is an API for converting HTML snippets containing any of 
MathJax input formats into HTML snippets containing SVG or MathML.

See the comments in the individual files for more details.

The `bin` directory contains a collection of command-line programs for 
converting among MathJax's various formats.  These can be used as examples 
of calling the MathJax API.

Use

    npm install mathjax-node

to install mathjax-node and its dependencies.

These API's can produce PNG images, but that requires the
[Batik](http://xmlgraphics.apache.org/batik/download.html) library.  It 
should be installed in the `batik` directory.  See the README file in that 
directory for more details.

# Getting started

mathjax-node provides two libraries, `./lib/mj-single.js` and `./lib/mj-page.js`. Below are two  very minimal examples -- be sure to check out the examples in `./bin/` for more advanced configurations.

* `./lib/mj-single.js` is optimized for processing single equations.


```javascript
// a simple TeX-input example
var mjAPI = require("mathjax-node/lib/mj-single.js");
mjAPI.config({
  MathJax: {
    // traditional MathJax configuration
  }
});
mjAPI.start();

var yourMath = 'E = mc^2';

mjAPI.typeset({
  math: yourMath,
  format: "TeX", // "inline-TeX", "MathML"
  mml:true, //  svg:true,
}, function (data) {
  if (!data.errors) {console.log(data.mml)}
});
```


* `./lib/mj-page.js` is optimized for handling full HTML pages. 


```javascript
var mjAPI = require("mathjax-node/lib/mj-page.js");
var jsdom = require("jsdom").jsdom;

var document = jsdom("<!DOCTYPE html><html lang='en'><head><title>Test</title></head><body><h1>Let's test mj-page</h1> <p> \\[f: X \\to Y\\], where \\( X = 2^{\mathbb{N}}\\) </p></body></html>");

mjAPI.start();

mjAPI.typeset({
  html: document.body.innerHTML,
  renderer: "NativeMML",
  inputs: ["TeX"],
  xmlns: "mml"
}, function(result) {
  "use strict";
  document.body.innerHTML = result.html;
  var HTML = "<!DOCTYPE html>\n" + document.documentElement.outerHTML.replace(/^(\n|\s)*/, "");
  console.log(HTML);
});
```
