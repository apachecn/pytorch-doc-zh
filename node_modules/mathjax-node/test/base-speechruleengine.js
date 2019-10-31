var tape = require('tape');
var mjAPI = require("..//lib/mj-single.js");

tape('basic test: check speechruleengine', function(t) {
  t.plan(1);

  var tex = 'MathJax';
  mjAPI.start();

  mjAPI.typeset({
    math: tex,
    format: "TeX",
    mml: true,
    speakText: true
  }, function(data) {
    t.ok(data.speakText, 'speechruleengine seems ok');
  });
});
