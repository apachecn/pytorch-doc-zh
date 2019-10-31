/*************************************************************
 *
 *  Copyright (c) 2018 The MathJax Consortium
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
 * @fileoverview  Implements the SVGmerror wrapper for the MmlMerror object
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {SVGWrapper, SVGConstructor} from '../Wrapper.js';
import {MmlMerror} from '../../../core/MmlTree/MmlNodes/merror.js';
import {StyleList} from '../../common/CssStyles.js';

/*****************************************************************/
/**
 *  The SVGmerror wrapper for the MmlMerror object
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 */
export class SVGmerror<N, T, D> extends SVGWrapper<N, T, D> {

    public static kind = MmlMerror.prototype.kind;

    public static styles: StyleList = {
        'g[data-mml-node="merror"] > g': {
            fill: 'red',
            stroke: 'red'
        },
        'g[data-mml-node="merror"] > rect[data-background]': {
            fill: 'yellow',
            stroke: 'none'
        }
    };

    toSVG(parent: N) {
        const svg = this.standardSVGnode(parent);
        const {h, d, w} = this.getBBox();
        this.adaptor.append(this.element, this.svg('rect', {
            'data-background': true,
            width: this.fixed(w), height: this.fixed(h + d), y: this.fixed(-d)
        }));
        this.addChildren(svg);
    }

}
