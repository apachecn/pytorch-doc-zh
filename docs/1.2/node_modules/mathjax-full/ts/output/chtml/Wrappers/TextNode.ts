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
 * @fileoverview  Implements the CHTMLTextNode wrapper for the TextNode object
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {TextNode} from '../../../core/MmlTree/MmlNode.js';
import {CHTMLWrapper, CHTMLConstructor} from '../Wrapper.js';
import {CommonTextNode, CommonTextNodeMixin} from '../../common/Wrappers/TextNode.js';
import {StyleList, StyleData} from '../../common/CssStyles.js';
import {OptionList} from '../../../util/Options.js';

/*****************************************************************/
/**
 *  The CHTMLTextNode wrapper for the TextNode object
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 */
export class CHTMLTextNode<N, T, D> extends CommonTextNodeMixin<CHTMLConstructor<any, any, any>>(CHTMLWrapper) {

    public static kind = TextNode.prototype.kind;

    public static autoStyle = false;

    public static styles: StyleList = {
        'mjx-c': {
            display: 'inline-block'
        },
        'mjx-utext': {
            display: 'inline-block',
            padding: '.75em 0 .25em 0'
        },
        'mjx-measure-text': {
            position: 'absolute',
            'font-family': 'MJXZERO',
            'white-space': 'nowrap',
            height: '1px',
            width: '1px',
            overflow: 'hidden'
        }
    };

    /**
     * @override
     */
    public toCHTML(parent: N) {
        this.markUsed();
        const adaptor = this.adaptor;
        const variant = this.parent.variant;
        const text = (this.node as TextNode).getText();
        if (variant === '-explicitFont') {
            const font = this.jax.getFontData(this.parent.styles);
            adaptor.append(parent, this.jax.unknownText(text, variant, font));
        } else {
            const c = this.parent.stretch.c;
            const chars = this.parent.remapChars(c ? [c] : this.unicodeChars(text));
            for (const n of chars) {
                const data = this.getVariantChar(variant, n)[3];
                const node = (data.unknown ?
                              this.jax.unknownText(String.fromCharCode(n), variant) :
                              this.html('mjx-c', {class: this.char(n)}));
                adaptor.append(parent, node);
                data.used = true;
            }
        }
    }

}
