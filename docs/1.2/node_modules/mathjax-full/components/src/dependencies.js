"use strict";
Object.defineProperty(exports, '__esModule', {value: true});

exports.dependencies = {
    'a11y/semantic-enrich': ['input/mml', '[sre]', 'input/mml'],
    'a11y/complexity': ['a11y/semantic-enrich'],
    'a11y/explorer': ['a11y/semantic-enrich', 'ui/menu'],
    '[tex]/all-packages': ['input/tex-base'],
    '[tex]/action': ['input/tex-base', '[tex]/newcommand'],
    '[tex]/autoload': ['input/tex-base', '[tex]/require'],
    '[tex]/ams': ['input/tex-base'],
    '[tex]/ams_cd': ['input/tex-base'],
    '[tex]/bbox': ['input/tex-base', '[tex]/ams', '[tex]/newcommand'],
    '[tex]/boldsymbol': ['input/tex-base'],
    '[tex]/braket': ['input/tex-base'],
    '[tex]/bussproofs': ['input/tex-base'],
    '[tex]/cancel': ['input/tex-base', '[tex]/enclose'],
    '[tex]/color': ['input/tex-base'],
    '[tex]/colorV2': ['input/tex-base'],
    '[tex]/configMacros': ['input/tex-base', '[tex]/newcommand'],
    '[tex]/enclose': ['input/tex-base'],
    '[tex]/extpfeil': ['input/tex-base', '[tex]/newcommand', '[tex]/ams'],
    '[tex]/html': ['input/tex-base'],
    '[tex]/mhchem': ['input/tex-base', '[tex]/ams'],
    '[tex]/newcommand': ['input/tex-base'],
    '[tex]/noerrors': ['input/tex-base'],
    '[tex]/noundefined': ['input/tex-base'],
    '[tex]/physics': ['input/tex-base'],
    '[tex]/require': ['input/tex-base'],
    '[tex]/tagFormat': ['input/tex-base'],
    '[tex]/unicode': ['input/tex-base'],
    '[tex]/verb': ['input/tex-base']
};

exports.paths = {
    tex: '[mathjax]/input/tex/extensions',
    sre: '[mathjax]/sre/sre_browser'
}

const allPackages = [
    '[tex]/action',
    '[tex]/ams',
    '[tex]/ams_cd',
    '[tex]/bbox',
    '[tex]/boldsymbol',
    '[tex]/braket',
    '[tex]/bussproofs',
    '[tex]/cancel',
    '[tex]/color',
    '[tex]/configMacros',
    '[tex]/enclose',
    '[tex]/extpfeil',
    '[tex]/html',
    '[tex]/mhchem',
    '[tex]/newcommand',
    '[tex]/noerrors',
    '[tex]/noundefined',
    '[tex]/physics',
    '[tex]/require',
    '[tex]/unicode',
    '[tex]/verb'
];

exports.provides = {
    'startup': ['loader'],
    'input/tex': [
        'input/tex-base',
        '[tex]/ams',
        '[tex]/newcommand',
        '[tex]/noundefined',
        '[tex]/require',
        '[tex]/autoload',
        '[tex]/configMacros'
    ],
    'input/tex-full': [
        'input/tex-base',
        '[tex]/all-packages',
        ...allPackages
    ],
    '[tex]/all-packages': allPackages
}
