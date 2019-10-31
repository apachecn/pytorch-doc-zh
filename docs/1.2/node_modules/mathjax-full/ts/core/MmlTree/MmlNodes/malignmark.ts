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
 * @fileoverview  Implements the MmlMalignmark node
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {PropertyList} from '../../Tree/Node.js';
import {AbstractMmlNode} from '../MmlNode.js';

/*****************************************************************/
/**
 *  Implements the MmlMalignmark node class (subclass of AbstractMmlNode)
 */

export class MmlMalignmark extends AbstractMmlNode {
    public static defaults: PropertyList = {
        ...AbstractMmlNode.defaults,
        edge: 'left'
    };

    /**
     * @return {string}  The malignmark kind
     */
    public get kind() {
        return 'malignmark';
    }

    /**
     * @return {number}  No children allowed
     */
    public get arity() {
        return 0;
    }

    /**
     * @return {boolen} <malignmark> is space-like
     */
    public get isSpacelike() {
        return true;
    }
}
