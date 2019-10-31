var Q = require('q');
var fs = require('fs');
var path = require('path');
var crc = require('crc');
var exec = require('child_process').exec;
var mjAPI = require('mathjax-node/lib/mj-single.js');

var started = false;
var countMath = 0;
var cache = {};

/**
    Prepare MathJaX
*/
function prepareMathJax() {
    if (started) {
        return;
    }

    mjAPI.config({
        MathJax: {
            SVG: {
                font: 'TeX'
            }
        }
    });
    mjAPI.start();

    started = true;
}

/**
    Convert a tex formula into a SVG text

    @param {String} tex
    @param {Object} options
    @return {Promise<String>}
*/
function convertTexToSvg(tex, options) {
    var d = Q.defer();
    options = options || {};

    prepareMathJax();

    mjAPI.typeset({
        math:           tex,
        format:         (options.inline ? 'inline-TeX' : 'TeX'),
        svg:            true,
        speakText:      true,
        speakRuleset:   'mathspeak',
        speakStyle:     'default',
        ex:             6,
        width:          100,
        linebreaks:     true
    }, function (data) {
        if (data.errors) {
            return d.reject(new Error(data.errors));
        }

        d.resolve(options.write? null : data.svg);
    });

    return d.promise;
}

/**
    Process a math block

    @param {Block} blk
    @return {Promise<Block>}
*/
function processBlock(blk) {
    var book = this;
    var tex = blk.body;
    var isInline = !(tex[0] == "\n");

    // For website return as script
    var config = book.config.get('pluginsConfig.mathjax', {});

    if ((book.output.name == "website" || book.output.name == "json")
        && !config.forceSVG) {
        return '<script type="math/tex; '+(isInline? "": "mode=display")+'">'+blk.body+'</script>';
    }

    // Check if not already cached
    var hashTex = crc.crc32(tex).toString(16);

    // Return
    var imgFilename = '_mathjax_' + hashTex + '.svg';
    var img = '<img src="/' + imgFilename + '" />';

    // Center math block
    if (!isInline) {
        img = '<div style="text-align:center;margin: 1em 0em;width: 100%;">' + img + '</div>';
    }

    return {
        body: img,
        post: function() {
            if (cache[hashTex]) {
                return;
            }

            cache[hashTex] = true;
            countMath = countMath + 1;

            return convertTexToSvg(tex, { inline: isInline })
            .then(function(svg) {
                return book.output.writeFile(imgFilename, svg);
            });
        }
    };
}

/**
    Return assets for website

    @return {Object}
*/
function getWebsiteAssets() {
    var version = this.config.get('pluginsConfig.mathjax.version', 'latest');

    return {
        assets: "./book",
        js: [
            'https://cdn.mathjax.org/mathjax/' + version + '/MathJax.js?config=TeX-AMS-MML_HTMLorMML',
            'plugin.js'
        ]
    };
}

module.exports = {
    website: getWebsiteAssets,
    blocks: {
        math: {
            shortcuts: {
                parsers: ["markdown", "asciidoc"],
                start: "$$",
                end: "$$"
            },
            process: processBlock
        }
    }
};
