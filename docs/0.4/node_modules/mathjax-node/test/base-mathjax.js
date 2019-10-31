var tape = require('tape');
var mjAPI = require("..//lib/mj-single.js");

tape('basic test: check MathJax core', function(t) {
  t.plan(1);

  var tex = '';
  mjAPI.start();

  mjAPI.typeset({
    math: tex,
    format: "TeX",
    mml: true
  }, function(data) {
    t.ok(data.mml, 'MathJax core seems ok');
  });
});
