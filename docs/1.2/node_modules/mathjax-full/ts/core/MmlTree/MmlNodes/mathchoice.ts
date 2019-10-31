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
 * @fileoverview  Implements the mathchoice node
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {PropertyList} from '../../Tree/Node.js';
import {AbstractMmlBaseNode, MmlNode, TEXCLASS, AttributeList} from '../MmlNode.js';

/*****************************************************************/
/**
 *  Implements the mathchoice node class (subclass of AbstractMmlBaseNode)
 *
 *  This is used by TeX's \mathchoice macro, but removes itself
 *  during the setInheritedAttributes process
 */

export class mathchoice extends AbstractMmlBaseNode {
    public static defaults: PropertyList = {
        ...AbstractMmlBaseNode.defaults
    };

    /**
     *  @return {string}  The mathcoice kind
     */
    public get kind() {
        return 'mathchoice';
    }

    /**
     *  @return {number}  4 children (display, text, script, and scriptscript styles)
     */
    public get arity() {
        return 4;
    }

    /**
     *  @return {boolean}  This element is not considered a MathML container
     */
    public get notParent() {
        return true;
    }

    /**
     * Replace the mathchoice node with the selected on based on the displaystyle and scriptlevel settings
     * (so the mathchoice never ends up in a finished MmlNode tree)
     *
     * @override
     */
    setInheritedAttributes(attributes: AttributeList, display: boolean, level: number, prime: boolean) {
        const selection = (display ? 0 : Math.max(0, Math.min(level, 2)) + 1);
        const child = this.childNodes[selection] || this.factory.create('mrow');
        this.parent.replaceChild(child, this);
        child.setInheritedAttributes(attributes, display, level, prime);
    }

}
