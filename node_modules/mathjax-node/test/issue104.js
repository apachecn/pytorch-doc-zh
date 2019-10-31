var tape = require('tape');
var mjAPI = require("..//lib/mj-single.js");
var jsdom = require('jsdom').jsdom;

tape('the SVG width should match the default', function(t) {
  t.plan(1);

  mjAPI.start();
  var tex = 'a \\\\ b';
  var expected = '100ex';

  mjAPI.typeset({
    math: tex,
    format: "TeX",
    svg: true
  }, function(data) {
    var document = jsdom(data.svg);
    var window = document.defaultView;
    var element = window.document.getElementsByTagName("svg")[0];
    var width = element.getAttribute('width');
    t.equal(width, expected);
  });
});
