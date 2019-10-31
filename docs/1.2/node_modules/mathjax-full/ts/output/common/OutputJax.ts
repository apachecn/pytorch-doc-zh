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
 * @fileoverview  Implements the abstract class for the CommonOutputJax
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {AbstractOutputJax} from '../../core/OutputJax.js';
import {MathDocument} from '../../core/MathDocument.js';
import {MathItem, Metrics} from '../../core/MathItem.js';
import {MmlNode} from '../../core/MmlTree/MmlNode.js';
import {FontData, FontDataClass, CharOptions, DelimiterData, CssFontData} from './FontData.js';
import {OptionList, separateOptions} from '../../util/Options.js';
import {CssStyles} from './CssStyles.js';
import {WrapperClass} from '../../core/Tree/Wrapper.js';
import {CommonWrapper, AnyWrapper, AnyWrapperClass} from './Wrapper.js';
import {CommonWrapperFactory, AnyWrapperFactory} from './WrapperFactory.js';
import {percent} from '../../util/lengths.js';
import {StyleList, Styles} from '../../util/Styles.js';
import {StyleList as CssStyleList} from './CssStyles.js';

/*****************************************************************/

/**
 * Maps linking a node to the test node it contains,
 *  and a map linking a node to the metrics within that node.
 */
export type MetricMap<N> = Map<N, Metrics>;
type MetricDomMap<N> = Map<N, N>;

/**
 * Maps for unknown characters
 */
export type UnknownBBox = {w: number, h: number, d: number};
export type UnknownMap = Map<string, UnknownBBox>;
export type UnknownVariantMap = Map<string, UnknownMap>;

/*****************************************************************/

/**
 *  The CommonOutputJax class on which the CHTML and SVG jax are built
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 * @template W  The Wrapper class
 * @template F  The WrapperFactory class
 * @template FD The FontData class
 * @template FC The FontDataClass object
 */
export abstract class CommonOutputJax<
    N, T, D,
    W extends AnyWrapper,
    F extends AnyWrapperFactory,
    FD extends FontData<any, any, any>,
    FC extends FontDataClass<any, any, any>
> extends AbstractOutputJax<N, T, D> {

    public static NAME: string = 'Common';

    public static OPTIONS: OptionList = {
        ...AbstractOutputJax.OPTIONS,
        scale: 1,                      // global scaling factor for all expressions
        minScale: .5,                  // smallest scaling factor to use
        matchFontHeight: true,         // true to match ex-height of surrounding font
        mtextInheritFont: false,       // true to make mtext elements use surrounding font
        merrorInheritFont: true,       // true to make merror text use surrounding font
        mathmlSpacing: false,          // true for MathML spacing rules, false for TeX rules
        skipAttributes: {},            // RFDa and other attributes NOT to copy to the output
        exFactor: .5,                  // default size of ex in em units
        displayAlign: 'center',        // default for indentalign when set to 'auto'
        displayIndent: '0',            // default for indentshift when set to 'auto'
        wrapperFactory: null,          // The wrapper factory to use
        font: null,                    // The FontData object to use
        cssStyles: null                // The CssStyles object to use
    };

    /**
     *  The default styles for the output jax
     */
    public static commonStyles: CssStyleList = {};

    /**
     * Used for collecting styles needed for the output jax
     */
    public cssStyles: CssStyles;

    /**
     * The MathDocument for the math we find,
     * the MathItem currently being processed,
     * and the container element for the math
     */
    public document: MathDocument<N, T, D>;
    public math: MathItem<N, T, D>;
    public container: N;

    /**
     * The pixels per em for the math item being processed
     */
    public pxPerEm: number;

    /**
     * The data for the font in use
     */
    public font: FD;

    public factory: F;

    /**
     * A map from the nodes in the expression currently being processed to the
     * wrapper nodes for them (used by functions like core() to locate the wrappers
     * from the core nodes)
     */
    public nodeMap: Map<MmlNode, W>;

    /**
     * Nodes used to test for metric data
     */
    public testInline: N;
    public testDisplay: N;

    /**
     * Cache of unknonw character bounding boxes for this element
     */
    protected unknownCache: UnknownVariantMap;

    /*****************************************************************/

    /**
     * Get the WrapperFactory and connect it to this output jax
     * Get the cssStyle and font objects
     *
     * @param {OptionList} options         The configuration options
     * @param(FontDataClass} defaultFont  The default FontData constructor
     * @constructor
     */
    constructor(options: OptionList = null,
                defaultFactory: typeof CommonWrapperFactory = null,
                defaultFont: FC = null) {
        const [jaxOptions, fontOptions] = separateOptions(options, defaultFont.OPTIONS);
        super(jaxOptions);
        this.factory = this.options.wrapperFactory ||
            new defaultFactory<CommonOutputJax<N, T, D, W, F, FD, FC>, W,
                               AnyWrapperClass, CharOptions, DelimiterData, FD>();
        this.factory.jax = this;
        this.cssStyles = this.options.cssStyles || new CssStyles();
        this.font = this.options.font || new defaultFont(fontOptions);
        this.unknownCache = new Map();
    }

    /*****************************************************************/

    /**
     * Save the math document
     * Create the mjx-container node
     * Create the DOM output for the root MathML math node in the container
     * Return the container node
     *
     * @override
     */
    public typeset(math: MathItem<N, T, D>, html: MathDocument<N, T, D>) {
        this.setDocument(html);
        let node = this.createNode();
        this.toDOM(math, node, html);
        return node;
    }

    /**
     * @return {N}  The container DOM node for the typeset math
     */
    protected createNode() {
        const jax = (this.constructor as typeof CommonOutputJax).NAME;
        return this.html('mjx-container', {'class': 'MathJax', jax: jax});
    }

    /**
     * @param {N} node   The container whose scale is to be set
     */
    protected setScale(node: N) {
        const scale = this.math.metrics.scale * this.options.scale;
        if (scale !== 1) {
            this.adaptor.setStyle(node, 'fontSize', percent(scale));
        }
    }

    /**
     * Save the math document, if any, and the math item
     * Set the document where HTML nodes will be created via the adaptor
     * Recursively set the TeX classes for the nodes
     * Set the scaling for the DOM node
     * Create the nodeMap (maps MathML nodes to corresponding wrappers)
     * Create the HTML output for the root MathML node in the container
     * Clear the nodeMape
     * Execute the post-filters
     *
     * @param {MathItem} math      The math item to convert
     * @param {N} node             The contaier to place the result into
     * @param {MathDocument} html  The document containing the math
     */
    public toDOM(math: MathItem<N, T, D>, node: N, html: MathDocument<N, T, D> = null) {
        this.setDocument(html);
        this.math = math;
        this.pxPerEm = math.metrics.ex / this.font.params.x_height;
        math.root.setTeXclass(null);
        this.setScale(node);
        this.nodeMap = new Map<MmlNode, W>();
        this.container = node;
        this.processMath(math.root, node);
        this.nodeMap = null;
        this.executeFilters(this.postFilters, math, html, node);
    }

    /**
     * This is the actual typesetting function supplied by the subclass
     *
     * @param {MmlNode} math   The intenral MathML node of the root math element to process
     * @param {N} node         The container node where the math is to be typeset
     */
    protected abstract processMath(math: MmlNode, node: N): void;

    /*****************************************************************/

    /**
     * @param {MathItem} math      The MathItem to get the bounding box for
     * @param {MathDocument} html  The MathDocument for the math
     */
    public getBBox(math: MathItem<N, T, D>, html: MathDocument<N, T, D>) {
        this.setDocument(html);
        this.math = math;
        math.root.setTeXclass(null);
        this.nodeMap = new Map<MmlNode, W>();
        let bbox = this.factory.wrap(math.root).getBBox();
        this.nodeMap = null;
        return bbox;
    }

    /*****************************************************************/

    /**
     * @override
     */
    public getMetrics(html: MathDocument<N, T, D>) {
        this.setDocument(html);
        const adaptor = this.adaptor;
        const maps = this.getMetricMaps(html);
        for (const math of html.math) {
            const map = maps[math.display ? 1 : 0];
            const {em, ex, containerWidth, lineWidth, scale} = map.get(adaptor.parent(math.start.node));
            math.setMetrics(em, ex, containerWidth, lineWidth, scale);
        }
    }

    /**
     * @param {N} node            The container node whose metrics are to be measured
     * @param {boolean} display   True if the metrics are for displayed math
     * @return {Metrics}          Object containing em, ex, containerWidth, etc.
     */
    public getMetricsFor(node: N, display: boolean) {
        const test = this.getTestElement(node, display);
        const metrics = this.measureMetrics(test);
        this.adaptor.remove(test);
        return metrics;
    }

    /**
     * Get a MetricMap for the math list
     *
     * @param {MathDocument} html  The math document whose math list is to be processed.
     * @return {MetricMap[]}       The node-to-metrics maps for all the containers that have math
     */
    protected getMetricMaps(html: MathDocument<N, T, D>) {
        const adaptor = this.adaptor;
        const domMaps = [new Map() as MetricDomMap<N>, new Map() as MetricDomMap<N>];
        //
        // Add the test elements all at once (so only one reflow)
        // Currently, we do one test for each container element for in-line and one for display math
        //   (since we need different techniques for the two forms to avoid a WebKit bug).
        //   This may need to be changed to handle floating elements better, since that has to be
        //   done at the location of the math itself, not necessarily the end of the container.
        //
        for (const math of html.math) {
            const node = adaptor.parent(math.start.node);
            const map = domMaps[math.display? 1 : 0];
            if (!map.has(node)) {
                map.set(node, this.getTestElement(node, math.display));
            }
        }
        //
        // Measure the metrics for all the mapped elements
        //
        const maps = [new Map() as MetricMap<N>, new Map() as MetricMap<N>];
        for (const i of maps.keys()) {
            for (const node of domMaps[i].keys()) {
                maps[i].set(node, this.measureMetrics(domMaps[i].get(node)));
            }
        }
        //
        // Remove the test elements
        //
        for (const i of maps.keys()) {
            for (const node of domMaps[i].values()) {
                adaptor.remove(node);
            }
        }
        return maps;
    }

    /**
     * @param {N} node    The math element to be measured
     * @return {N}        The test elements that were added
     */
    protected getTestElement(node: N, display: boolean) {
        const adaptor = this.adaptor;
        if (!this.testInline) {
            this.testInline = this.html('mjx-test', {style: {
                display:           'inline-block',
                width:             '100%',
                'font-style':      'normal',
                'font-weight':     'normal',
                'font-size':       '100%',
                'font-size-adjust':'none',
                'text-indent':     0,
                'text-transform':  'none',
                'letter-spacing':  'normal',
                'word-spacing':    'normal',
                overflow:          'hidden',
                height:            '1px',
                'margin-right':    '-1px'
            }}, [
                this.html('mjx-left-box', {style: {
                    display: 'inline-block',
                    width: 0,
                    'float': 'left'
                }}),
                this.html('mjx-ex-box', {style: {
                    position: 'absolute',
                    overflow: 'hidden',
                    width: '1px', height: '60ex'
                }}),
                this.html('mjx-right-box', {style: {
                    display: 'inline-block',
                    width: 0,
                    'float': 'right'
                }})
            ]);
            this.testDisplay = adaptor.clone(this.testInline);
            adaptor.setStyle(this.testDisplay, 'display', 'table');
            adaptor.setStyle(this.testDisplay, 'margin-right', '');
            adaptor.setStyle(adaptor.firstChild(this.testDisplay) as N, 'display', 'none');
            const right = adaptor.lastChild(this.testDisplay) as N;
            adaptor.setStyle(right, 'display', 'table-cell');
            adaptor.setStyle(right, 'width', '10000em');
            adaptor.setStyle(right, 'float', '');
        }
        return adaptor.append(node, adaptor.clone(display? this.testDisplay : this.testInline) as N) as N;
    }

    /**
     * @param {N} node    The test node to measure
     * @return {Metrics}  The metric data for the given node
     */
    protected measureMetrics(node: N) {
        const adaptor = this.adaptor;
        const em = adaptor.fontSize(node);
        const ex = (adaptor.nodeSize(adaptor.childNode(node, 1) as N)[1] / 60) || (em * this.options.exFactor);
        const containerWidth = (adaptor.getStyle(node, 'display') === 'table' ?
                                adaptor.nodeSize(adaptor.lastChild(node) as N)[0] - 1 :
                                adaptor.nodeBBox(adaptor.lastChild(node) as N).left -
                                adaptor.nodeBBox(adaptor.firstChild(node) as N).left - 2);
        const scale = Math.max(this.options.minScale,
                               this.options.matchFontHeight ? ex / this.font.params.x_height / em : 1);
        const lineWidth = 1000000;      // no linebreaking (otherwise would be a percentage of cwidth)
        return {em, ex, containerWidth, lineWidth, scale} as Metrics;
    }

    /*****************************************************************/

    /**
     * @override
     */
    public styleSheet(html: MathDocument<N, T, D>) {
        this.setDocument(html);
        //
        //  Start with the common styles
        //
        this.cssStyles.clear();
        this.cssStyles.addStyles((this.constructor as typeof CommonOutputJax).commonStyles);
        //
        // Gather the CSS from the classes
        //
        for (const kind of this.factory.getKinds()) {
            this.addClassStyles(this.factory.getNodeClass(kind));
        }
        //
        // Get the font styles
        //
        this.cssStyles.addStyles(this.font.styles);
        //
        // Create the stylesheet for the CSS
        //
        const sheet = this.html('style', {id: 'MJX-styles'}, [this.text('\n' + this.cssStyles.cssText + '\n')]);
        return sheet as N;
    }

    /**
     * @param {any} CLASS  The Wrapper class whose styles are to be added
     */
    protected addClassStyles(CLASS: typeof CommonWrapper) {
        this.cssStyles.addStyles(CLASS.styles);
    }

    /*****************************************************************/

    /**
     * @param {MathDocument} html  The document to be used
     */
    protected setDocument(html: MathDocument<N, T, D>) {
        if (html) {
            this.document = html;
            this.adaptor.document = html.document;
        }
    }

    /**
     * @param {string} type      The type of HTML node to create
     * @param {OptionList} def   The properties to set on the HTML node
     * @param {(N|T)[]} content  Array of child nodes to set for the HTML node
     * @param {string}           The namespace for the element
     * @return {N}               The newly created DOM tree
     */
    public html(type: string, def: OptionList = {}, content: (N | T)[] = [], ns?: string) {
        return this.adaptor.node(type, def, content, ns);
    }

    /**
     * @param {string} text  The text string for which to make a text node
     *
     * @return {HTMLElement}  A text node with the given text
     */
    public text(text: string) {
        return this.adaptor.text(text);
    }

    /**
     * @param {number} m    A number to be shown with a fixed number of digits
     * @param {number=} n   The number of digits to use
     * @return {string}     The formatted number
     */
    public fixed(m: number, n: number = 3) {
        if (Math.abs(m) < .0006) {
          return '0';
        }
        return m.toFixed(n).replace(/\.?0+$/, '');
    }

    /*****************************************************************/
    /*
     *  Methods for handling text that is not in the current MathJax font
     */

    /**
     * Create a DOM node for text from a specific CSS font, or that is
     *  not in the current MathJax font
     *
     * @param {string} text        The text to be displayed
     * @param {string} variant     The name of the variant for the text
     * @param {CssFontData} font   The style cssText string containing the font information
     * @return {N}                 The text element containing the text
     */
    public abstract unknownText(text: string, variant: string): N;

    /**
     * Measure text from a specific font, or that isn't in the MathJax font
     *
     * @param {string} text        The text to measure
     * @param {string} variant     The variant for the text
     * @param {CssFontData} font   The family, italic, and bold data for explicit fonts
     * @return {Object}            The width, height, and depth of the text (in ems)
     */
    public measureText(text: string, variant: string, font: CssFontData = ['', false, false]) {
        const node = this.unknownText(text, variant);
        if (variant === '-explicitFont') {
            const styles = this.cssFontStyles(font);
            this.adaptor.setAttributes(node, {style: styles});
        }
        return this.measureTextNodeWithCache(node, text, variant, font);
    }

    /**
     * Get the size of a text node, caching the result, and using
     *   a cached result, if there is one.
     *
     * @param {N} text         The text element to measure
     * @param {string} chars   The string contained in the text node
     * @return {Object}        The width, height and depth for the text
     */
    public measureTextNodeWithCache(text: N, chars: string, variant: string, font: CssFontData = ['', false, false]) {
        if (variant === '-explicitFont') {
            variant = [font[0], font[1] ? 'T' : 'F', font[2] ? 'T' : 'F', ''].join('-');
        }
        if (!this.unknownCache.has(variant)) {
            this.unknownCache.set(variant, new Map());
        }
        const map = this.unknownCache.get(variant);
        const cached = map.get(chars);
        if (cached) return cached;
        const bbox = this.measureTextNode(text);
        map.set(chars, bbox);
        return bbox;
    }

    /**
     * Measure the width of a text element by placing it in the page
     *  and looking up its size (fake the height and depth, since we can't measure that)
     *
     * @param {N} text            The text element to measure
     * @return {Object}           The width, height and depth for the text (in ems)
     */
    public abstract measureTextNode(text: N): UnknownBBox;

    /**
     * @param {CssFontData} font   The family, style, and weight for the given font
     * @param {Styles} styles      The style object to add the font data to
     * @return {Styles}            The modified (or initialized) style object
     */
    public cssFontStyles(font: CssFontData, styles: StyleList = {}) {
        const [family, italic, bold] = font;
        styles['font-family'] = family;
        if (italic) styles['font-style'] = 'italic';
        if (bold) styles['font-weight'] = 'bold';
        return styles;
    }

    /**
     * @param {Styles} styles   The style object to query
     * @return {CssFontData}    The family, italic, and boolean values
     */
    public getFontData(styles: Styles) {
        if (!styles) {
            styles = new Styles();
        }
        return [styles.get('font-family'),
                styles.get('font-style') === 'italic',
                styles.get('font-weight') === 'bold'] as CssFontData;
    }

}
