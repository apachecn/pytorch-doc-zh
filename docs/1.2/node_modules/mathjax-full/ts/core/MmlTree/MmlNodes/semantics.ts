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
 * @fileoverview  Implements the MmlSemantics, MmlAnnotation, and MmlAnnotationXML nodes
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {PropertyList} from '../../Tree/Node.js';
import {MmlNode, AbstractMmlNode, AbstractMmlBaseNode} from '../MmlNode.js';

/*****************************************************************/
/**
 *  Implements the MmlMroot node class (subclass of AbstractMmlBaseNode)
 */

export class MmlSemantics extends AbstractMmlBaseNode {
    public static defaults: PropertyList = {
        ...AbstractMmlBaseNode.defaults,
        definitionUrl: null,
        encoding: null
    };

    /**
     * @return {string}  The semantics kind
     */
    public get kind() {
        return 'semantics';
    }

    /**
     * @return {number}  <semantics> requires at least one node
     */
    public get arity() {
        return 1;
    }

    /**
     * @return {boolean}  Ignore <semantics> when looking for partent node
     */
    public get notParent() {
        return true;
    }

}

/*****************************************************************/
/**
 *  Implements the MmlMroot node class (subclass of AbstractMmlNode)
 */

export class MmlAnnotationXML extends AbstractMmlNode {
    public static defaults: PropertyList = {
        ...AbstractMmlNode.defaults,
        definitionUrl: null,
        encoding: null,
        cd: 'mathmlkeys',
        name: '',
        src: null
    };

    /**
     * @return {string}  The annotation-xml kind
     */
    public get kind() {
        return 'annotation-xml';
    }

    /**
     * Children are XMLNodes, so don't bother inheritting to them
     *
     * @override
     */
    protected setChildInheritedAttributes() {}
}

/*****************************************************************/
/**
 *  Implements the MmlMroot node class (subclass of MmlAnnotationXML)
 */

export class MmlAnnotation extends MmlAnnotationXML {
    public static defaults = {
        ...MmlAnnotationXML.defaults
    };
    public properties = {
        isChars: true
    };

    /**
     * @return {string}  The annotation-xml kind
     */
    public get kind() {
        return 'annotation';
    }
}
