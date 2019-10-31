var path = require('path');
var tester = require('gitbook-tester');
var assert = require('assert');

var pkg = require('../package.json');

describe('emphasize', function() {
    it('should correctly replace by span block', function() {
        return tester.builder()
            .withContent('#test me \n\nHello world. {% em %}highlight{% endem %}')
            .withLocalPlugin(path.join(__dirname, '..'))
            .withBookJson({
                gitbook: pkg.engines.gitbook,
                plugins: ['emphasize']
            })
            .create()
            .then(function(result) {
                assert.equal(result[0].content, '<h1 id="test-me">test me</h1>\n<p>Hello world. <span class="pg-emphasize pg-emphasize-yellow" style="">highlight</span></p>')
            });
    });

    it('should accept inline markdown', function() {
        return tester.builder()
            .withContent('#test me \n\nHello world. {% em %}highlight **bold**{% endem %}')
            .withLocalPlugin(path.join(__dirname, '..'))
            .withBookJson({
                gitbook: pkg.engines.gitbook,
                plugins: ['emphasize']
            })
            .create()
            .then(function(result) {
                assert.equal(result[0].content, '<h1 id="test-me">test me</h1>\n<p>Hello world. <span class="pg-emphasize pg-emphasize-yellow" style="">highlight <strong>bold</strong></span></p>')
            });
    });
});