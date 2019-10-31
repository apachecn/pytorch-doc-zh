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
 * @fileoverview  Implements the FontData class for character bbox data
 *                and stretchy delimiters.
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {OptionList} from '../../util/Options.js';
import {StyleList} from './CssStyles.js';

/****************************************************************************/

/**
 * The extra options allowed in a CharData array
 */
export interface CharOptions {
    ic?: number;                  // italic correction value
    sk?: number;                  // skew value
    unknown?: boolean;            // true if not found in the given variant
};

/****************************************************************************/

/**
 * Data about a character
 *   [height, depth, width, {italic-correction, skew, options}]
 *
 * @template C  The CharOptions type
 */
export type CharData<C extends CharOptions> =
    [number, number, number] |
    [number, number, number, C];

/**
 * An object making character positions to character data
 *
 * @template C  The CharOptions type
 */
export type CharMap<C extends CharOptions> = {
    [n: number]: CharData<C>;
};

/**
 * An object making variants to character maps
 *
 * @template C  The CharOptions type
 */
export type CharMapMap<C extends CharOptions> = {
    [name: string]: CharMap<C>;
};

/****************************************************************************/

/**
 * Data for a variant
 *
 * @template C  The CharOptions type
 */
export interface VariantData<C extends CharOptions> {
    /**
     * A list of CharMaps that must be updated when characters are
     * added to this variant
     */
    linked: CharMap<C>[];
    /**
     * The character data for this variant
     */
    chars: CharMap<C>;
};

/**
 * An object making variants names to variant data
 *
 * @template C  The CharOptions type
 * @template V  The VariantData type
 */
export type VariantMap<C extends CharOptions, V extends VariantData<C>> = {
    [name: string]: V;
};


/**
 * Data to use to map unknown characters in a variant to a
 * generic CSS font:
 *
 *    [fontname, italic, bold]
 */
export type CssFontData = [string, boolean, boolean];

/**
 * An object mapping a variant name to the CSS data needed for it
 */
export type CssFontMap = {
    [name: string]: CssFontData;
}

/****************************************************************************/

/**
 * Stretchy delimiter data
 */
export const enum DIRECTION {None, Vertical, Horizontal}
export const V = DIRECTION.Vertical;
export const H = DIRECTION.Horizontal;

/****************************************************************************/

/**
 * Data needed for stretchy vertical and horizontal characters
 */
export type DelimiterData = {
    dir: DIRECTION;       // vertical or horizontal direction
    sizes?: number[];     // Array of fixed sizes for this character
    variants?: number[];  // The variants in which the different sizes can be found (if not the default)
    schar?: number[];     // The character number to use for each size (if different from the default)
    stretch?: number[];   // The unicode code points for the parts of multi-character versions [beg, ext, end, mid?]
    HDW?: number[];       // [h, d, w] (for vertical, h and d are the normal size, w is the multi-character width,
                          //            for horizontal, h and d are the multi-character ones, w is for the normal size).
    min?: number;         // The minimum size a multi-character version can be
    c?: number;           // The character number (for aliased delimiters)
};

/**
 * An object mapping character numbers to delimiter data
 *
 * @template D  The DelimiterData type
 */
export type DelimiterMap<D extends DelimiterData> = {
    [n: number]: D;
};

/**
 * Delimiter data for a non-stretchy character
 */
export const NOSTRETCH: DelimiterData = {dir: DIRECTION.None};

/****************************************************************************/

/**
 * Data for remapping characters
 */
export type RemapData = string;
export type RemapMap = {
    [key: number]: RemapData;
}
export type RemapMapMap = {
    [key: string]: RemapMap;
}

/****************************************************************************/

/**
 * Font parameters (for TeX typesetting rules)
 */
export type FontParameters = {
    x_height: number,
    quad: number,
    num1: number,
    num2: number,
    num3: number,
    denom1: number,
    denom2: number,
    sup1: number,
    sup2: number,
    sup3: number,
    sub1: number,
    sub2: number,
    sup_drop: number,
    sub_drop: number,
    delim1: number,
    delim2: number,
    axis_height: number,
    rule_thickness: number,
    big_op_spacing1: number,
    big_op_spacing2: number,
    big_op_spacing3: number,
    big_op_spacing4: number,
    big_op_spacing5: number,

    surd_height: number,

    scriptspace: number,
    nulldelimiterspace: number,
    delimiterfactor: number,
    delimitershortfall: number,

    min_rule_thickness: number
};

/****************************************************************************/
/**
 *  The FontData class (for storing character bounding box data by variant,
 *                      and the stretchy delimiter data).
 *
 * @template C  The CharOptions type
 * @template V  The VariantData type
 * @template D  The DelimiterData type
 */
export class FontData<C extends CharOptions, V extends VariantData<C>, D extends DelimiterData> {

    /**
     * Subclasses may need options
     */
    public static OPTIONS: OptionList = {};

    /**
     *  The standard variants to define
     */
    public static defaultVariants = [
        ['normal'],
        ['bold', 'normal'],
        ['italic', 'normal'],
        ['bold-italic', 'italic', 'bold'],
        ['double-struck', 'bold'],
        ['fraktur', 'normal'],
        ['bold-fraktur', 'bold', 'fraktur'],
        ['script', 'normal'],
        ['bold-script', 'bold', 'script'],
        ['sans-serif', 'normal'],
        ['bold-sans-serif', 'bold', 'sans-serif'],
        ['sans-serif-italic', 'italic', 'sans-serif'],
        ['bold-sans-serif-italic', 'bold-italic', 'sans-serif'],
        ['monospace', 'normal']
    ];

    /**
     * The style and weight to use for each variant (for unkown characters)
     */
    public static defaultCssFonts: CssFontMap = {
        normal: ['serif', false, false],
        bold: ['serif', false, true],
        italic: ['serif', true, false],
        'bold-italic': ['serif', true, true],
        'double-struck': ['serif', false, true],
        fraktur: ['serif', false, false],
        'bold-fraktur': ['serif', false, true],
        script: ['cursive', false, false],
        'bold-script': ['cursive', false, true],
        'sans-serif': ['sans-serif', false, false],
        'bold-sans-serif': ['sans-serif', false, true],
        'sans-serif-italic': ['sans-serif', true, false],
        'bold-sans-serif-italic': ['sans-serif', true, true],
        monospace: ['monospace', false, false]
    };

    /**
     *  The default remappings
     */
    protected static defaultAccentMap = {
        0x0300: '\u02CB',  // grave accent
        0x0301: '\u02CA',  // acute accent
        0x0302: '\u02C6',  // curcumflex
        0x0303: '\u02DC',  // tilde accent
        0x0304: '\u02C9',  // macron
        0x0306: '\u02D8',  // breve
        0x0307: '\u02D9',  // dot
        0x0308: '\u00A8',  // diaresis
        0x030A: '\u02DA',  // ring above
        0x030C: '\u02C7',  // caron
        0x2192: '\u20D7',
        0x2032: '\'',
        0x2033: '\'\'',
        0x2034: '\'\'\'',
        0x2035: '`',
        0x2036: '``',
        0x2037: '```',
        0x2057: '\'\'\'\'',
        0x20D0: '\u21BC', // combining left harpoon
        0x20D1: '\u21C0', // combining right harpoon
        0x20D6: '\u2190', // combining left arrow
        0x20E1: '\u2194', // combining left-right arrow
        0x20F0: '*',      // combining asterisk
        0x20DB: '...',    // combining three dots above
        0x20DC: '....',   // combining four dots above
        0x20EC: '\u21C1', // combining low left harpoon
        0x20ED: '\u21BD', // combining low right harpoon
        0x20EE: '\u2190', // combining low left arrows
        0x20EF: '\u2192'  // combining low right arrows
    };

    protected static defaultMoMap = {
        0x002D: '\u2212' // hyphen
    };

    protected static defaultMnMap = {
        0x002D: '\u2212' // hyphen
    }

    /**
     *  The default font parameters for the font
     */
    public static defaultParams: FontParameters = {
        x_height:         .442,
        quad:             1,
        num1:             .676,
        num2:             .394,
        num3:             .444,
        denom1:           .686,
        denom2:           .345,
        sup1:             .413,
        sup2:             .363,
        sup3:             .289,
        sub1:             .15,
        sub2:             .247,
        sup_drop:         .386,
        sub_drop:         .05,
        delim1:          2.39,
        delim2:          1.0,
        axis_height:      .25,
        rule_thickness:   .06,
        big_op_spacing1:  .111,
        big_op_spacing2:  .167,
        big_op_spacing3:  .2,
        big_op_spacing4:  .6,
        big_op_spacing5:  .1,

        surd_height:      .075,

        scriptspace:         .05,
        nulldelimiterspace:  .12,
        delimiterfactor:     901,
        delimitershortfall:   .3,

        min_rule_thickness:  1.25     // in pixels
    };

    /**
     * The default delimiter and character data
     */
    protected static defaultDelimiters: DelimiterMap<any> = {};
    protected static defaultChars: CharMapMap<any> = {};

    /**
     * The default variants for the fixed size stretchy delimiters
     */
    protected static defaultSizeVariants: string[] = [];

    /**
     * @param {CharMap} font   The font to check
     * @param {number} n       The character to get options for
     * @return {CharOptions}   The options for the character
     */
    public static charOptions(font: CharMap<CharOptions>, n: number) {
        const char = font[n];
        if (char.length === 3) {
            (char as any)[3] = {};
        }
        return char[3];
    }

    /**
     * The actual variant, delimiter, and size information for this font
     */
    protected variant: VariantMap<C, V> = {};
    protected delimiters: DelimiterMap<D> = {};
    protected sizeVariants: string[];
    protected cssFontMap: CssFontMap = {};

    /**
     * The character maps
     */
    protected remapChars: RemapMapMap = {};

    /**
     * The actual font parameters for this font
     */
    public params: FontParameters;

    /**
     * Any styles needed for the font
     */
    public styles: StyleList;

    /**
     * Copies the data from the defaults to the instance
     *
     * @constructor
     */
    constructor() {
        let CLASS = (this.constructor as typeof FontData);
        this.params = {...CLASS.defaultParams};
        this.sizeVariants = [...CLASS.defaultSizeVariants];
        this.cssFontMap = {...CLASS.defaultCssFonts};
        this.createVariants(CLASS.defaultVariants);
        this.defineDelimiters(CLASS.defaultDelimiters);
        for (const name of Object.keys(CLASS.defaultChars)) {
            this.defineChars(name, CLASS.defaultChars[name]);
        }
        this.defineRemap('accent', CLASS.defaultAccentMap);
        this.defineRemap('mo', CLASS.defaultMoMap);
        this.defineRemap('mn', CLASS.defaultMnMap);
    }

    /**
     * Creates the data structure for a variant -- an object with
     *   prototype chain that includes a copy of the linked variant,
     *   and then the inherited variant chain.
     *
     *   The reason for this extra link is that for a mathvariant like
     *   bold-italic, you want to inherit from both the bold and
     *   italic variants, but the prototype chain can only inherit
     *   from one. So for bold-italic, we make an object that has a
     *   prototype consisting of a copy of the bold data, and add the
     *   italic data as the prototype chain. (Since this is a copy, we
     *   keep a record of this link so that if bold is changed later,
     *   we can update this copy. That is not needed for the prototype
     *   chain, since the prototypes are the actual objects, not
     *   copies.) We then use this bold-plus-italic object as the
     *   prototype chain for the bold-italic object
     *
     *   That means that bold-italic will first look in its own object
     *   for specifically bold-italic glyphs that are defined there,
     *   then in the copy of the bold glyphs (only its top level is
     *   copied, not its prototype chain), and then the specifically
     *   italic glyphs, and then the prototype chain for italics,
     *   which is the normal glyphs. Effectively, this means
     *   bold-italic looks for bold-italic, then bold, then italic,
     *   then normal glyphs in order to find the given character.
     *
     * @param {string} name     The new variant to create
     * @param {string} inherit  The variant to use if a character is not in this one
     * @param {string} link     A variant to search before the inherit one (but only
     *                           its top-level object).
     */
    public createVariant(name: string, inherit: string = null, link: string = null) {
        let variant = {
            linked: [] as CharMap<C>[],
            chars: (inherit ? Object.create(this.variant[inherit].chars) : {}) as CharMap<C>
        } as V;
        if (link && this.variant[link]) {
            Object.assign(variant.chars, this.variant[link].chars);
            this.variant[link].linked.push(variant.chars);
            variant.chars = Object.create(variant.chars);
        }
        this.variant[name] = variant;
    }

    /**
     * Create a collection of variants
     *
     * @param {string[][]} variants  Array of [name, inherit?, link?] values for
     *                              the variants to define
     */
    public createVariants(variants: string[][]) {
        for (const variant of variants) {
            this.createVariant(variant[0], variant[1], variant[2]);
        }
    }

    /**
     * Defines new character data in a given variant
     *  (We use Object.assign() here rather than the spread operator since
     *  the character maps are objeccts with prototypes, and we don't
     *  want to loose those by doing {...chars} or something similar.)
     *
     * @param {string} name    The variant for these characters
     * @param {CharMap} chars  The characters to define
     */
    public defineChars(name: string, chars: CharMap<C>) {
        let variant = this.variant[name];
        Object.assign(variant.chars, chars);
        for (const link of variant.linked) {
            Object.assign(link, chars);
        }
    }

    /**
     * Defines stretchy delimiters
     *
     * @param {DelimiterMap} delims  The delimiters to define
     */
    public defineDelimiters(delims: DelimiterMap<D>) {
        Object.assign(this.delimiters, delims);
    }

    /**
     * Defines a character remapping map
     *
     * @param {string} name     The name of the map to define or augment
     * @param {RemapMap} remap  The characters to remap
     */
    public defineRemap(name: string, remap: RemapMap) {
        if (!this.remapChars.hasOwnProperty(name)) {
            this.remapChars[name] = {};
        }
        Object.assign(this.remapChars[name], remap);
    }

    /**
     * @param {number} n  The delimiter character number whose data is desired
     * @return {DelimiterData}  The data for that delimiter (or undefined)
     */
    public getDelimiter(n: number) {
        return this.delimiters[n];
    }

    /**
     * @param {number} n  The delimiter character number whose variant is needed
     * @param {number} i  The index in the size array of the size whose variant is needed
     * @return {string}   The variant of the i-th size for delimiter n
     */
    public getSizeVariant(n: number, i: number) {
        if (this.delimiters[n].variants) {
            i = this.delimiters[n].variants[i];
        }
        return this.sizeVariants[i];
    }

    /**
     * @param {string} name  The variant whose character data is being querried
     * @param {number} n     The unicode number for the character to be found
     * @return {CharData}    The data for the given character (or undefined)
     */
    public getChar(name: string, n: number) {
        return this.variant[name].chars[n];
    }

    /**
     * @param {string} name   The name of the variant whose data is to be obtained
     * @return {VariantData}  The data for the requested variant (or undefined)
     */
    public getVariant(name: string) {
        return this.variant[name];
    }

    /**
     * @param {string} variant   The name of the variant whose data is to be obtained
     * @return {CssFontData}     The CSS data for the requested variant
     */
    public getCssFont(variant: string) {
        return this.cssFontMap[variant] || ['serif', false, false];
    }

    /**
     * @param {string} name   The name of the map to query
     * @param {number} c      The character to remap
     * @return {number}       The remapped character (or the original)
     */
    public getRemappedChar(name: string, c: number) {
        const map = this.remapChars[name] || {} as RemapMap;
        return map[c];
    }

}

/**
 * The class interface for the FontData class
 *
 * @template C  The CharOptions type
 * @template V  The VariantData type
 * @template D  The DelimiterData type
 */
export interface FontDataClass<C extends CharOptions, V extends VariantData<C>, D extends DelimiterData> {
    OPTIONS: OptionList;
    defaultCssFonts: CssFontMap;
    defaultVariants: string[][];
    defaultParams: FontParameters;
    charOptions(font: CharMap<C>, n: number): C
    new(...args: any[]): FontData<C, V, D>;
};
