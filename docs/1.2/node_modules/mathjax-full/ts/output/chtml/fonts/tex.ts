/*************************************************************
 *
 *  Copyright (c) 2017 The MathJax Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/**
 * @fileoverview  The MathJax TeXFont object
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {CHTMLFontData, CHTMLCharOptions, CHTMLCharData, CHTMLVariantData, CHTMLDelimiterData, CHTMLFontDataClass,
        CssFontMap, DelimiterData, DelimiterMap, CharMapMap, FontDataClass} from '../FontData.js';
import {CommonTeXFontMixin} from '../../common/fonts/tex.js';
import {StringMap} from '../Wrapper.js';

import {boldItalic} from './tex/bold-italic.js';
import {bold} from './tex/bold.js';
import {doubleStruck} from './tex/double-struck.js';
import {frakturBold} from './tex/fraktur-bold.js';
import {fraktur} from './tex/fraktur.js';
import {italic} from './tex/italic.js';
import {largeop} from './tex/largeop.js';
import {monospace} from './tex/monospace.js';
import {normal} from './tex/normal.js';
import {sansSerifBoldItalic} from './tex/sans-serif-bold-italic.js';
import {sansSerifBold} from './tex/sans-serif-bold.js';
import {sansSerifItalic} from './tex/sans-serif-italic.js';
import {sansSerif} from './tex/sans-serif.js';
import {scriptBold} from './tex/script-bold.js';
import {script} from './tex/script.js';
import {smallop} from './tex/smallop.js';
import {texCalligraphicBold} from './tex/tex-calligraphic-bold.js';
import {texCalligraphic} from './tex/tex-calligraphic.js';
import {texMathit} from './tex/tex-mathit.js';
import {texOldstyleBold} from './tex/tex-oldstyle-bold.js';
import {texOldstyle} from './tex/tex-oldstyle.js';
import {texSize3} from './tex/tex-size3.js';
import {texSize4} from './tex/tex-size4.js';
import {texVariant} from './tex/tex-variant.js';

import {delimiters} from '../../common/fonts/tex/delimiters.js';

/*=================================================================================*/
/**
 *  The TeXFont class
 */
export class TeXFont extends
CommonTeXFontMixin<CHTMLCharOptions, CHTMLVariantData, CHTMLDelimiterData, CHTMLFontDataClass>(CHTMLFontData) {

    /**
     * The classes to use for each variant
     */
    protected static defaultVariantClasses: StringMap = {
        'normal': 'mjx-n',
        'bold': 'mjx-b',
        'italic': 'mjx-i',
        'bold-italic': 'mjx-b mjx-i',
        'double-struck': 'mjx-ds',
        'fraktur': 'mjx-fr',
        'bold-fraktur': 'mjx-fr mjx-b',
        'script': 'mjx-sc',
        'bold-script': 'mjx-sc mjx-b',
        'sans-serif': 'mjx-ss',
        'bold-sans-serif': 'mjx-ss mjx-b',
        'sans-serif-italic': 'mjx-ss mjx-i',
        'bold-sans-serif-italic': 'mjx-ss mjx-b mjx-i',
        'monospace': 'mjx-ty',
        '-smallop': 'mjx-sop',
        '-largeop': 'mjx-lop',
        '-size3': 'mjx-s3',
        '-size4': 'mjx-s4',
        '-tex-calligraphic': 'mjx-cal',
        '-tex-bold-calligraphic': 'mjx-cal mjx-b',
        '-tex-mathit': 'mjx-mit',
        '-tex-oldstyle': 'mjx-os',
        '-tex-bold-oldstyle': 'mjx-os mjx-b',
        '-tex-variant': 'mjx-v'
    };

    /**
     *  The stretchy delimiter data
     */
    protected static defaultDelimiters: DelimiterMap<CHTMLDelimiterData> = delimiters;

    /**
     *  The character data by variant
     */
    protected static defaultChars: CharMapMap<CHTMLCharOptions> = {
        'normal': normal,
        'bold': bold,
        'italic': italic,
        'bold-italic': boldItalic,
        'double-struck': doubleStruck,
        'fraktur': fraktur,
        'bold-fraktur': frakturBold,
        'script': script,
        'bold-script': scriptBold,
        'sans-serif': sansSerif,
        'bold-sans-serif': sansSerifBold,
        'sans-serif-italic': sansSerifItalic,
        'bold-sans-serif-italic': sansSerifBoldItalic,
        'monospace': monospace,
        '-smallop': smallop,
        '-largeop': largeop,
        '-size3': texSize3,
        '-size4': texSize4,
        '-tex-calligraphic': texCalligraphic,
        '-tex-bold-calligraphic': texCalligraphicBold,
        '-tex-mathit': texMathit,
        '-tex-oldstyle': texOldstyle,
        '-tex-bold-oldstyle': texOldstyleBold,
        '-tex-variant': texVariant
    };

    /*=====================================================*/
    /**
     * The CSS styles needed for this font.
     */
    protected static defaultStyles = {
        ...CHTMLFontData.defaultStyles,

        '.mjx-n mjx-c': {
            'font-family': 'MJXZERO, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-i mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-b.mjx-i mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-BI, MJXTEX-B, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-cal mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-C, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-cal.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-C-B, MJXTEX-C, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-ds mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-A, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1'
        },

        '.mjx-fr mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-FR, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-fr.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-FR-B, MJXTEX-FR, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-sc mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SC, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-sc.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SC-B, MJXTEX-SC, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-ss mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SS, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-ss.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SS-B, MJXTEX-SS, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-ss.mjx-i mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SS-I, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-ss.mjx-b.mjx-i mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-SS-B, MJXTEX-SS-I, MJXTEX-BI, MJXTEX-B, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-ty mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-T, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-var mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-A, MJXTEX, MJXTEX-I, MJXTEX-S1'
        },

        '.mjx-os mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-C, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },
        '.mjx-os.mjx-b mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-C-B, MJXTEX-C, MJXTEX-B, MJXTEX-BI, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-mit mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-MI, MJXTEX-I, MJXTEX, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-lop mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-S2, MJXTEX-S1, MJXTEX, MJXTEX-I, MJXTEX-A'
        },

        '.mjx-sop mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-S1, MJXTEX, MJXTEX-I, MJXTEX-A'
        },

        '.mjx-s3 mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-S3, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.mjx-s4 mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-S4, MJXTEX, MJXTEX-I, MJXTEX-S1, MJXTEX-A'
        },

        '.MJX-TEX': {
            'font-family': 'MJXZERO'
        },

        'mjx-stretchy-v mjx-c, mjx-stretchy-h mjx-c': {
            'font-family': 'MJXZERO, MJXTEX-S1, MJXTEX-S4, MJXTEX, MJXTEX-A ! important'
        }
    };

    /**
     * The default @font-face declarations with %%URL%% where the font path should go
     */
    protected static defaultFonts = {
        ...CHTMLFontData.defaultFonts,

        '@font-face /* 1 */': {
            'font-family': 'MJXTEX',
            src: 'url("%%URL%%/MathJax_Main-Regular.woff") format("woff")'
        },

        '@font-face /* 2 */': {
            'font-family': 'MJXTEX-B',
            src: 'url("%%URL%%/MathJax_Main-Bold.woff") format("woff")'
        },

        '@font-face /* 3 */': {
            'font-family': 'MJXTEX-MI',
            src: 'url("%%URL%%/MathJax_Main-Italic.woff") format("woff")'
        },

        '@font-face /* 4 */': {
            'font-family': 'MJXTEX-I',
            src: 'url("%%URL%%/MathJax_Math-Italic.woff") format("woff")'
        },

        '@font-face /* 5 */': {
            'font-family': 'MJXTEX-BI',
            src: 'url("%%URL%%/MathJax_Math-BoldItalic.woff") format("woff")'
        },

        '@font-face /* 6 */': {
            'font-family': 'MJXTEX-S1',
            src: 'url("%%URL%%/MathJax_Size1-Regular.woff") format("woff")'
        },

        '@font-face /* 7 */': {
            'font-family': 'MJXTEX-S2',
            src: 'url("%%URL%%/MathJax_Size2-Regular.woff") format("woff")'
        },

        '@font-face /* 8 */': {
            'font-family': 'MJXTEX-S3',
            src: 'url("%%URL%%/MathJax_Size3-Regular.woff") format("woff")'
        },

        '@font-face /* 9 */': {
            'font-family': 'MJXTEX-S4',
            src: 'url("%%URL%%/MathJax_Size4-Regular.woff") format("woff")'
        },

        '@font-face /* 10 */': {
            'font-family': 'MJXTEX-A',
            src: 'url("%%URL%%/MathJax_AMS-Regular.woff") format("woff")'
        },

        '@font-face /* 11 */': {
            'font-family': 'MJXTEX-C',
            src: 'url("%%URL%%/MathJax_Calligraphic-Regular.woff") format("woff")'
        },

        '@font-face /* 12 */': {
            'font-family': 'MJXTEX-C-B',
            src: 'url("%%URL%%/MathJax_Calligraphic-Bold.woff") format("woff")'
        },

        '@font-face /* 13 */': {
            'font-family': 'MJXTEX-FR',
            src: 'url("%%URL%%/MathJax_Fraktur-Regular.woff") format("woff")'
        },

        '@font-face /* 14 */': {
            'font-family': 'MJXTEX-FR-B',
            src: 'url("%%URL%%/MathJax_Fraktur-Bold.woff") format("woff")'
        },

        '@font-face /* 15 */': {
            'font-family': 'MJXTEX-SS',
            src: 'url("%%URL%%/MathJax_SansSerif-Regular.woff") format("woff")'
        },

        '@font-face /* 16 */': {
            'font-family': 'MJXTEX-SS-B',
            src: 'url("%%URL%%/MathJax_SansSerif-Bold.woff") format("woff")'
        },

        '@font-face /* 17 */': {
            'font-family': 'MJXTEX-SS-I',
            src: 'url("%%URL%%/MathJax_SansSerif-Italic.woff") format("woff")'
        },

        '@font-face /* 18 */': {
            'font-family': 'MJXTEX-SC',
            src: 'url("%%URL%%/MathJax_Script-Regular.woff") format("woff")'
        },

        '@font-face /* 19 */': {
            'font-family': 'MJXTEX-T',
            src: 'url("%%URL%%/MathJax_Typewriter-Regular.woff") format("woff")'
        },

        '@font-face /* 20 */': {
            'font-family': 'MJXTEX-V',
            src: 'url("%%URL%%/MathJax_Vector-Regular.woff") format("woff")'
        },

        '@font-face /* 21 */': {
            'font-family': 'MJXTEX-VB',
            src: 'url("%%URL%%/MathJax_Vector-Bold.woff") format("woff")'
        },
    };

}
