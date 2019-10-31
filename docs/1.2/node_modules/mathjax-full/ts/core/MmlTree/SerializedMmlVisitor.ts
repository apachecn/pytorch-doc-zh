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
 * @fileoverview  A visitor that produces a serilaied MathML string
 *                (replacement for toMathML() output from v2)
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {MmlVisitor} from './MmlVisitor.js';
import {MmlFactory} from './MmlFactory.js';
import {MmlNode, TextNode, XMLNode, TEXCLASSNAMES} from './MmlNode.js';

/*****************************************************************/
/**
 *  Implements the SerializedMmlVisitor (subclass of MmlVisitor)
 */

export class SerializedMmlVisitor extends MmlVisitor {

    /**
     * Convert the tree rooted at a particular node into a serialized MathML string
     *
     * @param {MmlNode} node  The node to use as the root of the tree to traverse
     * @return {string}       The MathML string representing the internal tree
     */
    public visitTree(node: MmlNode) {
        return this.visitNode(node, '');
    }

    /**
     * @param {TextNode} node  The text node to visit
     * @param {string} space   The amount of indenting for this node
     * @return {string}        The (HTML-quoted) text of the node
     */
    public visitTextNode(node: TextNode, space: string) {
        return this.quoteHTML(node.getText());
    }

    /**
     * @param {XMLNode} node  The XML node to visit
     * @param {string} space  The amount of indenting for this node
     * @return {string}       The serialization of the XML node (not implemented yet).
     */
    public visitXMLNode(node: XMLNode, space: string) {
        return '[XML Node not implemented]';
    }

    /**
     * Visit an inferred mrow, but don't add the inferred row itself (since
     * it is supposed to be inferred).
     *
     * @param {MmlNode} node  The inferred mrow to visit
     * @param {string} space  The amount of indenting for this node
     * @return {string}       The serialized contents of the mrow, properly indented
     */
    public visitInferredMrowNode(node: MmlNode, space: string) {
        let mml = [];
        for (const child of node.childNodes) {
            mml.push(this.visitNode(child, space));
        }
        return mml.join('\n');
    }

    /**
     * Visit a TeXAtom node. It is turned into a mrow with the appropriate TeX class
     * indicator.
     *
     * @param {MmlNode} node  The TeXAtom to visit.
     * @param {string} space  The amount of indenting for this node.
     * @return {string}       The serialized contents of the mrow, properly indented.
     */
    public visitTeXAtomNode(node: MmlNode, space: string) {
      let texclass = node.texClass < 0 ? 'NONE' : TEXCLASSNAMES[node.texClass];
      let mml = space + '<mrow class="MJX-TeXAtom-' + texclass + '"' +
          this.getAttributes(node) + '>\n';
      const endspace = space;
      space += '  ';
      for (const child of node.childNodes) {
        mml += this.visitNode(child, space);
      }
      mml += '\n' + endspace + '</mrow>';
      return mml;
    }

    /**
     * @param {MmlNode} node    The annotation node to visit
     * @param {string} space    The number of spaces to use for indentation
     * @return {string}         The serializied annotation element
     */
    public visitAnnotationNode(node: MmlNode, space: string) {
        return space + '<annotation' + this.getAttributes(node) + '>'
             + this.childNodeMml(node, '', '')
             + '</annotation>';
    }

    /**
     * The generic visiting function:
     *   Make the string versino of the open tag, properly indented, with it attributes
     *   Increate the indentation level
     *   Add the childnodes
     *   Add the end tag with proper spacing (empty tags have the close tag following directly)
     *
     * @param {MmlNode} node    The node to visit
     * @param {Element} parent  The DOM parent to which this node should be added
     * @return {string}         The serialization of the given node
     */
    public visitDefault(node: MmlNode, space: string) {
        let kind = node.kind;
        let [nl, endspace] = (node.isToken || node.childNodes.length === 0 ? ['', ''] : ['\n', space]);
        const children = this.childNodeMml(node, space + '  ', nl);
        return space + '<' + kind + this.getAttributes(node) + '>'
            + (children.match(/\S/) ? nl + children + endspace : '')
            + '</' + kind + '>';
    }

    /**
     * @param {MmlNode} node    The node whose children are to be added
     * @param {string} space    The spaces to use for indentation
     * @param {string} nl       The newline character (or empty)
     * @return {string}         The serializied children
     */
    protected childNodeMml(node: MmlNode, space: string, nl: string) {
        let mml = '';
        for (const child of node.childNodes) {
            mml += this.visitNode(child, space) + nl;
        }
        return mml;
    }

    /**
     * @param {MmlNode} node  The node whose attributes are to be produced
     * @return {string}       The attribute list as a string
     */
    protected getAttributes(node: MmlNode) {
        let ATTR = '';
        let attributes = node.attributes.getAllAttributes();
        for (const name of Object.keys(attributes)) {
            if (attributes[name] === undefined) continue;
            ATTR += ' ' + name + '="' + this.quoteHTML(attributes[name].toString()) + '"';
        }
        return ATTR;
    }

    /**
     *  Convert HTML special characters to entities (&amp;, &lt;, &gt;, &quot;)
     *  Convert multi-character Unicode characters to entities
     *  Convert non-ASCII characters to entities.
     *
     * @param {string} value  The string to be made HTML escaped
     * @return {string}       The string with escaping performed
     */
    protected quoteHTML(value: string) {
        return value
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\"/g, '&quot;')
            .replace(/([\uD800-\uDBFF].)/g, (m, c) => {
                return '&#x' + ((c.charCodeAt(0) - 0xD800) * 0x400 +
                                (c.charCodeAt(1) - 0xDC00) + 0x10000).toString(16).toUpperCase() + ';';
            })
            .replace(/([\u0080-\uD7FF\uE000-\uFFFF])/g, (m, c) => {
                return '&#x' + c.charCodeAt(0).toString(16).toUpperCase() + ';';
            });
    }

}
