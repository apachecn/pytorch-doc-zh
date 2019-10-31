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
 * @fileoverview  Implements the MmlMaligngroup node
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {PropertyList} from '../../Tree/Node.js';
import {AbstractMmlNode, AttributeList} from '../MmlNode.js';
import {INHERIT} from '../Attributes.js';

/*****************************************************************/
/**
 *  Implements the MmlMaligngroup node class (subclass of AbstractMmlNode)
 */

export class MmlMaligngroup extends AbstractMmlNode {
    public static defaults: PropertyList = {
        ...AbstractMmlNode.defaults,
        groupalign: INHERIT
    };

    /**
     * @return {string}  The maligngroup kind
     */
    public get kind() {
        return 'maligngroup';
    }

    /**
     * @return {boolean}  <maligngroup> is space-like
     */
    public get isSpacelike() {
        return true;
    }

    /**
     * Children can inherit from <maligngroup>
     *
     * @override
     */
    protected setChildInheritedAttributes(attributes: AttributeList, display: boolean, level: number, prime: boolean) {
        attributes = this.addInheritedAttributes(attributes, this.attributes.getAllAttributes());
        super.setChildInheritedAttributes(attributes, display, level, prime);
    }
}
