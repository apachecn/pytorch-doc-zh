import './lib/asciimath.js';

import {AsciiMath} from '../../../../js/input/asciimath.js';

if (MathJax.startup) {
    MathJax.Hub.Config({AsciiMath: MathJax.config.asciimath || {}});
    MathJax.startup.registerConstructor('asciimath', AsciiMath);
    MathJax.startup.useInput('asciimath');
}
