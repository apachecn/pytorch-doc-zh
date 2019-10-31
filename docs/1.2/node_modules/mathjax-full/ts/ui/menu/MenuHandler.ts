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
 * @fileoverview  Mixin that adds a context-menu to MathJax output
 *
 * @author dpvc@mathjax.org (Davide Cervone)
 */

import {mathjax} from '../../mathjax.js';

import {MathItem, STATE, newState} from '../../core/MathItem.js';
import {MathDocumentConstructor} from '../../core/MathDocument.js';
import {HTMLDocument} from '../../handlers/html/HTMLDocument.js';
import {Handler} from '../../core/Handler.js';
import {ComplexityMathDocument, ComplexityMathItem} from '../../a11y/complexity.js';
import {ExplorerMathDocument, ExplorerMathItem} from '../../a11y/explorer.js';
import {OptionList, expandable} from '../../util/Options.js';

import {Menu} from './Menu.js';

/*==========================================================================*/

/**
 * Generic constructor for Mixins
 */
export type Constructor<T> = new(...args: any[]) => T;

/**
 * Constructor for base MathItem for MenuMathItem
 */
export type A11yMathItemConstructor = {
    new(...args: any[]): ComplexityMathItem<HTMLElement, Text, Document> & ExplorerMathItem;
}

/**
 * Constructor for base document for MenuMathDocument
 */
export type A11yDocumentConstructor =
    MathDocumentConstructor<ComplexityMathDocument<HTMLElement, Text, Document> & ExplorerMathDocument>;

/*==========================================================================*/

/**
 * Add STATE value for menus being added (after TYPESET and before INSERTED)
 */
newState('CONTEXT_MENU', 170);

/**
 * The new function for MathItem that adds the context menu
 */
export interface MenuMathItem extends ComplexityMathItem<HTMLElement, Text, Document> {

    /**
     * @param {MenuMathDocument} document   The document where the menu is being added
     */
    addMenu(document: MenuMathDocument): void;

    /**
     * @param {MenuMathDocument} document   The document where the menu is being added
     * @param {boolean=} force              True if enrichment should not depend on the menu setting
     */
    enrich(document: MenuMathDocument, force?: boolean): void;

    /**
     * @param {MenuMathDocument} document   The document where the menu is being added
     * @param {boolean=} force              True if complexity computation should not depend on the menu setting
     */
    complexity(document: MenuMathDocument, force?: boolean): void;

    /**
     * @param {MenuMathDocument} document   The document where the menu is being added
     * @param {boolean=} force              True if exploration should not depend on the menu setting
     */
    explorable(document: MenuMathDocument, force?: boolean): void;

    /**
     * @param {MenuMathDocument} document   The document to check for if anything is being loaded
     */
    checkLoading(document: MenuMathDocument): void;

}

/**
 * The mixin for adding context menus to MathItems
 *
 * @param {B} BaseMathItem   The MathItem class to be extended
 * @return {MathMathItem}    The extended MathItem class
 *
 * @template B  The MathItem class to extend
 */
export function MenuMathItemMixin<B extends A11yMathItemConstructor>(
    BaseMathItem: B
): Constructor<MenuMathItem> & B {

    return class extends BaseMathItem {

        /**
         * @param {MenuMathDocument} document   The document where the menu is being added
         */
        public addMenu(document: MenuMathDocument) {
            if (this.state() < STATE.CONTEXT_MENU) {
                document.menu.addMenu(this);
                this.state(STATE.CONTEXT_MENU);
            }
        }

        /**
         * @param {MenuMathDocument} document   The document to check for if anything is being loaded
         */
        public checkLoading(document: MenuMathDocument) {
            if (document.menu.isLoading) {
                mathjax.retryAfter(document.menu.loadingPromise.catch((err) => console.log(err)));
            }
        }

        /**
         * @override
         */
        public enrich(document: MenuMathDocument, force: boolean = false) {
            const settings = document.menu.settings;
            if (settings.collapsible || settings.explorer || force) {
                settings.collapsible && document.menu.checkComponent('a11y/complexity');
                settings.explorer    && document.menu.checkComponent('a11y/explorer');
                super.enrich(document);
            }
        }

        /**
         * @override
         */
        public complexity(document: MenuMathDocument, force: boolean = false) {
            if (document.menu.settings.collapsible || force) {
                document.menu.checkComponent('a11y/complexity');
                super.complexity(document);
            }
        }

        /**
         * @override
         */
        public explorable(document: MenuMathDocument, force: boolean = false) {
            if (document.menu.settings.explorer || force) {
                document.menu.checkComponent('a11y/explorer');
                super.explorable(document);
            }
        }

    }
}

/*==========================================================================*/

/**
 * The properties needed in the MathDocument for context menus
 */
export interface MenuMathDocument extends ComplexityMathDocument<HTMLElement, Text, Document> {

    /**
     * The menu associated with this document
     */
    menu: Menu;

    /**
     * Add context menus to the MathItems in the MathDocument
     *
     * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
     */
    addMenu(): MenuMathDocument;

    /**
     * @param {boolean=} force      True if enrichment should not depend on the menu settings
     * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
     */
    enrich(force?: boolean): MenuMathDocument;

    /**
     * @param {boolean=} force      True if complexity computation should not depend on the menu settings
     * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
     */
    complexity(force?: boolean): MenuMathDocument;

    /**
     * @param {boolean=} force      True if exploration should not depend on the menu settings
     * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
     */
    explorable(force?: boolean): MenuMathDocument;

    /**
     * Checks if there are files being loaded by the menu, and restarts the typesetting if so
     *
     * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
     */
    checkLoading(): MenuMathDocument;
}

/**
 * The mixin for adding context menus to MathDocuments
 *
 * @param {B} BaseMathDocument     The MathDocument class to be extended
 * @return {MenuMathDocument}      The extended MathDocument class
 *
 * @template B  The MathDocument class to extend
 */
export function MenuMathDocumentMixin<B extends A11yDocumentConstructor>(
    BaseDocument: B
): Constructor<MenuMathDocument> & B {

    return class extends BaseDocument {

        /**
         * @override
         */
        public static OPTIONS = {
            ...BaseDocument.OPTIONS,
            MenuClass: Menu,
            menuOptions: Menu.OPTIONS,
            a11y: (BaseDocument.OPTIONS.a11y || expandable({})),
            renderActions: expandable({
                ...BaseDocument.OPTIONS.renderActions,
                addMenu: [STATE.CONTEXT_MENU],
                checkLoading: [STATE.UNPROCESSED + 1]
            })
        }

        /**
         * The menu associated with this document
         */
        public menu: Menu;

        /**
         * Extend the MathItem class used for this MathDocument
         *
         * @override
         * @constructor
         */
        constructor(...args: any[]) {
            super(...args);
            this.menu = new this.options.MenuClass(this, this.options.menuOptions);
            const ProcessBits = (this.constructor as typeof BaseDocument).ProcessBits;
            if (!ProcessBits.has('context-menu')) {
                ProcessBits.allocate('context-menu');
            }
            this.options.MathItem = MenuMathItemMixin<A11yMathItemConstructor>(this.options.MathItem);
        }

        /**
         * Add context menus to the MathItems in the MathDocument
         *
         * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
         */
        public addMenu() {
            if (!this.processed.isSet('context-menu')) {
                for (const math of this.math) {
                    (math as MenuMathItem).addMenu(this);
                }
                this.processed.set('context-menu');
            }
            return this;
        }

        /**
         * Checks if there are files being loaded by the menu, and restarts the typesetting if so
         *
         * @return {MenuMathDocument}   The MathDocument (so calls can be chained)
         */
        public checkLoading() {
            if (this.menu.isLoading) {
                mathjax.retryAfter(this.menu.loadingPromise.catch((err) => console.log(err)));
            }
            return this;
        }

        /**
         * @override
         */
        public state(state: number, restore: boolean = false) {
            super.state(state, restore);
            if (state < STATE.CONTEXT_MENU) {
                this.processed.clear('context-menu');
            }
            return this;
        }

        /**
         * @override
         */
        public updateDocument() {
            super.updateDocument();
            (this.menu.menu.getStore() as any).sort();
            return this;
        }

        /**
         * @param {boolean=} force       True if enrichment should not depend on menu settings
         * @returns {MenuMathDocument}   The MathDocument (for chaining of calls)
         */
        public enrich(force: boolean = false) {
            const settings = this.menu.settings;
            if (!this.processed.isSet('enriched') && (settings.collapsible || settings.explorer || force)) {
                settings.collapsible && this.menu.checkComponent('a11y/complexity');
                settings.explorer    && this.menu.checkComponent('a11y/explorer');
                for (const math of this.math) {
                    (math as MenuMathItem).enrich(this, force);
                }
                this.processed.set('enriched');
            }
            return this;
        }

        /**
         * @param {boolean=} force       True if complexity computations should not depend on menu settings
         * @returns {MenuMathDocument}   The MathDocument (for chaining of calls)
         */
        public complexity(force: boolean = false) {
            if (!this.processed.isSet('complexity') && (this.menu.settings.collapsible || force)) {
                this.menu.checkComponent('a11y/complexity');
                for (const math of this.math) {
                    (math as MenuMathItem).complexity(this, force);
                }
                this.processed.set('complexity');
            }
            return this;
        }

        /**
         * @param {boolean=} force       True if exploration should not depend on menu settings
         * @returns {MenuMathDocument}   The MathDocument (for chaining of calls)
         */
        public explorable(force: boolean = false) {
            if (!this.processed.isSet('explorer') && (this.menu.settings.explorer || force)) {
                this.menu.checkComponent('a11y/explorer');
                for (const math of this.math) {
                    (math as MenuMathItem).explorable(this, force);
                }
                this.processed.set('explorer');
            }
            return this;
        }

    };

}

/*==========================================================================*/

/**
 * Add context-menu support to a Handler instance
 *
 * @param {Handler} handler   The Handler instance to enhance
 * @return {Handler}          The handler that was modified (for purposes of chaining extensions)
 */
export function MenuHandler(handler: Handler<HTMLElement, Text, Document>) {
    handler.documentClass = MenuMathDocumentMixin<A11yDocumentConstructor>(handler.documentClass as any);
    return handler;
}
