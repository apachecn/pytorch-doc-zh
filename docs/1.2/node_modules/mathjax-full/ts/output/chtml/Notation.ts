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
 * @fileoverview  Implements utilities for notations for menclose elements
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {BBox} from './BBox.js';
import {CHTMLmenclose} from './Wrappers/menclose.js';
import * as Notation from '../common/Notation.js';
export * from '../common/Notation.js';

/**
 * Create a named element (handled by CSS), and adjust it if thickness is non-standard
 *
 * @param {string} name    The name of the element to create
 * @param {string} offset  The offset direction to adjust if thickness is non-standard
 * @return {Renderer}      The renderer function for the given element name
 */
export const RenderElement = <N, T, D>(name: string, offset: string = '') => {
    return ((node, child) => {
        const shape = node.adjustBorder(node.html('mjx-' + name));
        if (offset && node.thickness !== Notation.THICKNESS) {
            const transform = 'translate' + offset + '(' + node.em(node.thickness / 2) + ')';
            node.adaptor.setStyle(shape, 'transform', transform);
        }
        node.adaptor.append(node.chtml, shape);
    }) as Notation.Renderer<CHTMLmenclose<N, T, D>, N>;
};

/**
 * @param {string} side   The side on which a border should appear
 * @return {DefPair}      The notation definition for the notation having a line on the given side
 */
export const Border = <N, T, D>(side: Notation.Side) =>
    Notation.CommonBorder<CHTMLmenclose<N, T, D>, N>((node, child) => {
        node.adaptor.setStyle(child, 'border-' + side, node.em(node.thickness) + ' solid');
    })(side);


/**
 * @param {string} name    The name of the notation to define
 * @param {string} side1   The first side to get a border
 * @param {string} side2   The second side to get a border
 * @return {DefPair}       The notation definition for the notation having lines on two sides
 */
export const Border2 = <N, T, D>(name: string, side1: Notation.Side, side2: Notation.Side) =>
    Notation.CommonBorder2<CHTMLmenclose<N, T, D>, N>((node, child) => {
        const border = node.em(node.thickness) + ' solid';
        node.adaptor.setStyle(child, 'border-' + side1, border);
        node.adaptor.setStyle(child, 'border-' + side2, border);
    })(name, side1, side2);

/**
 * @param {string} name  The name of the diagonal strike to define
 * @param {number} neg   1 or -1 to use with the angle
 * @return {DefPair}     The notation definition for the diagonal strike
 */
export const DiagonalStrike = <N, T, D>(name: string, neg: number) =>
    Notation.CommonDiagonalStrike<CHTMLmenclose<N, T, D>, N>((cname: string) => (node, child) => {
        const {w, h, d} = node.getBBox();
        const [a, W] = node.getArgMod(w, h + d);
        const t = neg * node.thickness / 2;
        const strike = node.adjustBorder(node.html(cname, {style: {
            width: node.em(W),
            transform: 'rotate(' + node.fixed(-neg * a) + 'rad) translateY(' + t + 'em)',
        }}));
        node.adaptor.append(node.chtml, strike);
    })(name);

/**
 * @param {string} name   The name of the diagonal arrow to define
 * @return {DefPair}      The notation definition for the diagonal arrow
 */
export const DiagonalArrow = <N, T, D>(name: string) =>
    Notation.CommonDiagonalArrow<CHTMLmenclose<N, T, D>, N>((node, arrow) => {
        node.adaptor.append(node.chtml, arrow);
    })(name);

/**
 * @param {string} name   The name of the horizontal or vertical arrow to define
 * @return {DefPair}      The notation definition for the arrow
 */
export const Arrow = <N, T, D>(name: string) =>
    Notation.CommonArrow<CHTMLmenclose<N, T, D>, N>((node, arrow) => {
        node.adaptor.append(node.chtml, arrow);
    })(name);
