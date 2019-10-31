/*************************************************************
 *
 *  Copyright (c) 2019 The MathJax Consortium
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
 * @fileoverview  Implements the CommonMtext wrapper mixin for the MmlMtext object
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {AnyWrapper, WrapperConstructor, Constructor} from '../Wrapper.js';
import {BBox} from '../BBox.js';

/*****************************************************************/
/**
 * The CommonMtext interface
 */
export interface CommonMtext extends AnyWrapper {
}

/**
 * Shorthand for the CommonMtext constructor
 */
export type MtextConstructor = Constructor<CommonMtext>;

/*****************************************************************/
/**
 *  The CommonMtext wrapper mixin for the MmlMtext object
 *
 * @template T  The Wrapper class constructor type
 */
export function CommonMtextMixin<T extends WrapperConstructor>(Base: T): MtextConstructor & T {
    return class extends Base {

        /**
         * The font-family, weight, and style to use for the variants when mtextInheritFont
         * is true.  If not in this list, then usual MathJax fonts are used instead.
         */
        public static INHERITFONTS = {
            normal: ['', '', ''],
            bold: ['', 'bold', ''],
            italic: ['', '', 'italic'],
            'bold-italic': ['', 'bold', 'italic'],
            script: ['cursive', '', ''],
            'bold-script': ['cursive', 'bold', ''],
            'sans-serif': ['sans-serif', '', ''],
            'bold-sans-serif': ['sans-serif', 'bold', ''],
            'sans-serif-italic': ['sans-serif', '', 'italic'],
            'bold-sans-serif-italic': ['sans-serif', 'bold', 'italic'],
            monospace: ['monospace', '', '']
        };

        /**
         * @override
         */
        protected getVariant() {
            const options = this.jax.options;
            //
            //  If the font is to be inherited from the surrounding text, check the mathvariant
            //  and see if it allows for inheritance. If so, set the variant appropriately,
            //  otherwise get the usual variant.
            //
            if (options.mtextInheritFont || (options.merrorInheritFont && this.node.Parent.isKind('merror'))) {
                const variant = this.node.attributes.get('mathvariant') as string;
                const font = (this.constructor as any).INHERITFONTS[variant] as [string, string, string];
                if (font) {
                    this.variant = this.explicitVariant(...font);
                    return;
                }
            }
            super.getVariant();
        }

    };

}
