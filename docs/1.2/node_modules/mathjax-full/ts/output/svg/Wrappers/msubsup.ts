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
 * @fileoverview  Implements the SVGmsubsup wrapper for the MmlMsubsup object
 *                and the special cases SVGmsub and SVGmsup
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {SVGWrapper, SVGConstructor, Constructor} from '../Wrapper.js';
import {SVGscriptbase} from './scriptbase.js';
import {CommonMsub, CommonMsubMixin} from '../../common/Wrappers/msubsup.js';
import {CommonMsup, CommonMsupMixin} from '../../common/Wrappers/msubsup.js';
import {CommonMsubsup, CommonMsubsupMixin} from '../../common/Wrappers/msubsup.js';
import {MmlMsubsup, MmlMsub, MmlMsup} from '../../../core/MmlTree/MmlNodes/msubsup.js';

/*****************************************************************/
/**
 * The SVGmsub wrapper for the MmlMsub object
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 */
export class SVGmsub<N, T, D> extends
CommonMsubMixin<SVGWrapper<any, any, any>, Constructor<SVGscriptbase<any, any, any>>>(SVGscriptbase)  {

    public static kind = MmlMsub.prototype.kind;

    public static useIC = false;

}

/*****************************************************************/
/**
 * The SVGmsup wrapper for the MmlMsup object
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 */
export class SVGmsup<N, T, D> extends
CommonMsupMixin<SVGWrapper<any, any, any>, Constructor<SVGscriptbase<any, any, any>>>(SVGscriptbase)  {

    public static kind = MmlMsup.prototype.kind;

    public static useIC = true;

}

/*****************************************************************/
/**
 * The SVGmsubsup wrapper for the MmlMsubsup object
 *
 * @template N  The HTMLElement node class
 * @template T  The Text node class
 * @template D  The Document class
 */
export class SVGmsubsup<N, T, D> extends
CommonMsubsupMixin<SVGWrapper<any, any, any>, Constructor<SVGscriptbase<any, any, any>>>(SVGscriptbase)  {

    public static kind = MmlMsubsup.prototype.kind;

    public static useIC = false;

    /**
     * @override
     */
    public toSVG(parent: N) {
        const svg = this.standardSVGnode(parent);
        const [base, sup, sub] = [this.baseChild, this.supChild, this.subChild];
        const bbox = base.getBBox();
        const cbox = this.baseCore.bbox;
        const [u, v] = this.getUVQ(bbox, sub.getBBox(), sup.getBBox());

        base.toSVG(svg);
        sup.toSVG(svg);
        sub.toSVG(svg);

        sub.place(bbox.w, v);
        sup.place(bbox.w + this.coreIC(), u);
    }

}
