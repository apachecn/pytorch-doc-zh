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
 * @fileoverview  Implements the CommonMsqrt wrapper for the MmlMsqrt object
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {AnyWrapper, WrapperConstructor, Constructor} from '../Wrapper.js';
import {BBox} from '../BBox.js';
import {MmlMsqrt} from '../../../core/MmlTree/MmlNodes/msqrt.js';
import {DIRECTION} from '../FontData.js';

/*****************************************************************/
/**
 * The CommonMsqrt interface
 */
export interface CommonMsqrt extends AnyWrapper {
    /**
     * The index of the base of the root in childNodes
     */
    readonly base: number;

    /**
     * The index of the surd in childNodes
     */
    readonly surd: number;

    /**
     * The index of the root in childNodes (or null if none)
     */
    readonly root: number;

    /**
     * The requested height of the stretched surd character
     */
    surdH: number;

    /**
     * Combine the bounding box of the root (overridden in mroot)
     *
     * @param {BBox} bbox  The bounding box so far
     * @param {BBox} sbox  The bounding box of the surd
     */
    combineRootBBox(bbox: BBox, sbox: BBox): void;

    /**
     * @param {BBox} sbox  The bounding box for the surd character
     * @return {number[]}  The p, q, and x values for the TeX layout computations
     */
    getPQ(sbox: BBox): number[];

    /**
     * @param {BBox} sbox  The bounding box of the surd
     * @return {number[]}  The x offset of the surd, and the height, x offset, and scale of the root
     */
    getRootDimens(sbox: BBox): number[];

}

/**
 * Shorthand for the CommonMsqrt constructor
 */
export type MsqrtConstructor = Constructor<CommonMsqrt>;

/*****************************************************************/
/**
 * The CommonMsqrt wrapper mixin for the MmlMsqrt object
 *
 * @template T  The Wrapper class constructor type
 */
export function CommonMsqrtMixin<T extends WrapperConstructor>(Base: T): MsqrtConstructor & T {
    return class extends Base {

        /**
         * @return {number}  The index of the base of the root in childNodes
         */
        get base() {
            return 0;
        }

        /**
         * @return {number}  The index of the surd in childNodes
         */
        get surd() {
            return 1;
        }

        /**
         * @return {number}  The index of the root in childNodes (or null if none)
         */
        get root(): number {
            return null;
        }

        /**
         * The requested height of the stretched surd character
         */
        public surdH: number;

        /**
         * Add the surd character so we can display it later
         *
         * @override
         */
        constructor(...args: any[]) {
            super(...args);
            const surd = this.createMo('\u221A');
            surd.canStretch(DIRECTION.Vertical);
            const {h, d} = this.childNodes[this.base].getBBox();
            const t = this.font.params.rule_thickness;
            const p = (this.node.attributes.get('displaystyle') ? this.font.params.x_height : t);
            this.surdH = h + d + 2 * t + p / 4;
            surd.getStretchedVariant([this.surdH - d, d], true);
        }

        /**
         * @override
         */
        public createMo(text: string) {
            const node = super.createMo(text);
            this.childNodes.push(node);
            return node;
        }

        /**
         * @override
         */
        public computeBBox(bbox: BBox, recompute: boolean = false) {
            const surdbox = this.childNodes[this.surd].getBBox();
            const basebox = new BBox(this.childNodes[this.base].getBBox());
            const [p, q] = this.getPQ(surdbox);
            const [x] = this.getRootDimens(surdbox);
            const t = this.font.params.rule_thickness;
            const H = basebox.h + q + t;
            bbox.h = H + t;
            this.combineRootBBox(bbox, surdbox);
            bbox.combine(surdbox, x, H - surdbox.h);
            bbox.combine(basebox, x + surdbox.w, 0);
            bbox.clean();
            this.setChildPWidths(recompute);
        }

        /**
         * Combine the bounding box of the root (overridden in mroot)
         *
         * @param {BBox} bbox  The bounding box so far
         * @param {BBox} sbox  The bounding box of the surd
         */
        public combineRootBBox(bbox: BBox, sbox: BBox) {
        }

        /**
         * @param {BBox} sbox  The bounding box for the surd character
         * @return {number[]}  The p, q, and x values for the TeX layout computations
         */
        public getPQ(sbox: BBox) {
            const t = this.font.params.rule_thickness;
            const p = (this.node.attributes.get('displaystyle') ? this.font.params.x_height : t);
            const q = (sbox.h + sbox.d > this.surdH ? ((sbox.h + sbox.d) - (this.surdH - t)) / 2 : t + p / 4);
            return [p, q];
        }

        /**
         * @param {BBox} sbox  The bounding box of the surd
         * @return {number[]}  The x offset of the surd, and the height, x offset, and scale of the root
         */
        public getRootDimens(sbox: BBox) {
            return [0, 0, 0, 0];
        }

    };

}
