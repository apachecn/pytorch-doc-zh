declare namespace ContextMenu {
    interface Element {
        getHtml(): HTMLElement;
        setHtml(element: HTMLElement): void;
        generateHtml(): void;
    }
}
declare namespace ContextMenu {
    interface Item extends Entry, Element {
        content: string;
        getId(): string;
        press(): void;
        focus(): void;
        unfocus(): void;
        disable(): void;
        enable(): void;
    }
}
declare namespace ContextMenu {
    interface Postable {
        isPosted(): boolean;
        post(): void;
        post(x?: number, y?: number): void;
        unpost(): void;
    }
}
declare namespace ContextMenu {
    interface VariableItem {
        register(): void;
        unregister(): void;
        update(): void;
    }
}
declare namespace ContextMenu {
    class Variable<T> {
        private name;
        private getter;
        private setter;
        private items;
        constructor(name: string, getter: (node?: HTMLElement) => T, setter: (x: T, node?: HTMLElement) => void);
        getName(): string;
        getValue(node?: HTMLElement): T;
        setValue(value: T, node?: HTMLElement): void;
        register(item: VariableItem): void;
        unregister(item: VariableItem): void;
        update(): void;
        registerCallback(func: Function): void;
        unregisterCallback(func: Function): void;
    }
}
declare namespace ContextMenu {
    class VariablePool<T> {
        private pool;
        insert(variable: Variable<T>): void;
        lookup(name: string): Variable<T>;
        remove(name: string): void;
        update(): void;
    }
}
declare namespace ContextMenu {
    interface Menu extends Postable {
        getItems(): Item[];
        getFocused(): Item;
        setFocused(item?: Item): void;
        getPool(): VariablePool<string | boolean>;
        unpostSubmenus(): void;
        find(id: string): Item;
        generateMenu(): void;
    }
}
declare namespace ContextMenu {
    interface Entry {
        getMenu(): Menu;
        setMenu(menu: Menu): void;
        getType(): string;
        hide(): void;
        show(): void;
        isHidden(): boolean;
    }
}
declare namespace ContextMenu {
    enum KEY {
        RETURN = 13,
        ESCAPE = 27,
        SPACE = 32,
        LEFT = 37,
        UP = 38,
        RIGHT = 39,
        DOWN = 40,
    }
    interface KeyNavigatable {
        keydown(event: KeyboardEvent): void;
        escape(event: KeyboardEvent): void;
        space(event: KeyboardEvent): void;
        left(event: KeyboardEvent): void;
        right(event: KeyboardEvent): void;
        up(event: KeyboardEvent): void;
        down(event: KeyboardEvent): void;
    }
}
declare namespace ContextMenu {
    const MOUSE: {
        CLICK: string;
        DBLCLICK: string;
        DOWN: string;
        UP: string;
        OVER: string;
        OUT: string;
        MOVE: string;
        SELECTSTART: string;
        SELECTEND: string;
    };
    interface MouseNavigatable {
        mousedown(event: MouseEvent): void;
        mouseup(event: MouseEvent): void;
        mouseover(event: MouseEvent): void;
        mouseout(event: MouseEvent): void;
        click(event: MouseEvent): void;
    }
}
declare namespace ContextMenu {
    abstract class AbstractNavigatable implements KeyNavigatable, MouseNavigatable {
        private bubble;
        bubbleKey(): void;
        keydown(event: KeyboardEvent): void;
        escape(event: KeyboardEvent): void;
        space(event: KeyboardEvent): void;
        left(event: KeyboardEvent): void;
        right(event: KeyboardEvent): void;
        up(event: KeyboardEvent): void;
        down(event: KeyboardEvent): void;
        protected stop(event: Event): void;
        mousedown(event: MouseEvent): void;
        mouseup(event: MouseEvent): void;
        mouseover(event: MouseEvent): void;
        mouseout(event: MouseEvent): void;
        click(event: MouseEvent): void;
        addEvents(element: HTMLElement): void;
    }
}
declare namespace ContextMenu {
    const HtmlClasses: {
        [id: string]: HtmlClass;
    };
    type HtmlClass = 'CtxtMenu_ContextMenu' | 'CtxtMenu_Menu' | 'CtxtMenu_MenuArrow' | 'CtxtMenu_MenuActive' | 'CtxtMenu_MenuCheck' | 'CtxtMenu_MenuClose' | 'CtxtMenu_MenuDisabled' | 'CtxtMenu_MenuItem' | 'CtxtMenu_MenuLabel' | 'CtxtMenu_MenuRadioCheck' | 'CtxtMenu_MenuRule' | 'CtxtMenu_MousePost' | 'CtxtMenu_RTL' | 'CtxtMenu_Attached' | 'CtxtMenu_Info' | 'CtxtMenu_InfoClose' | 'CtxtMenu_InfoContent' | 'CtxtMenu_InfoSignature' | 'CtxtMenu_InfoTitle' | 'CtxtMenu_MenuFrame' | 'CtxtMenu_MenuInputBox';
    const HtmlAttrs: {
        [id: string]: HtmlAttr;
    };
    type HtmlAttr = 'CtxtMenu_Counter' | 'CtxtMenu_keydownFunc' | 'CtxtMenu_contextmenuFunc' | 'CtxtMenu_touchstartFunc' | 'CtxtMenu_Oldtabindex';
}
declare namespace ContextMenu {
    abstract class MenuElement extends AbstractNavigatable implements Element {
        protected role: string;
        protected className: HtmlClass;
        private html;
        addAttributes(attributes: {
            [attr: string]: string;
        }): void;
        getHtml(): HTMLElement;
        setHtml(html: HTMLElement): void;
        generateHtml(): void;
        focus(): void;
        unfocus(): void;
    }
}
declare namespace ContextMenu {
    abstract class AbstractEntry extends MenuElement implements Entry {
        protected className: HtmlClass;
        protected role: string;
        private menu;
        private type;
        private hidden;
        constructor(menu: Menu, type: string);
        getMenu(): Menu;
        setMenu(menu: Menu): void;
        getType(): string;
        hide(): void;
        show(): void;
        isHidden(): boolean;
    }
}
declare namespace ContextMenu {
    abstract class AbstractPostable extends MenuElement implements Postable {
        private posted;
        isPosted(): boolean;
        post(x?: number, y?: number): void;
        unpost(): void;
        protected abstract display(): void;
    }
}
declare namespace ContextMenu {
    abstract class AbstractMenu extends AbstractPostable implements Menu {
        protected className: HtmlClass;
        protected variablePool: VariablePool<string | boolean>;
        protected role: string;
        private items;
        private focused;
        getItems(): Item[];
        getPool(): VariablePool<string | boolean>;
        getFocused(): Item;
        setFocused(item: Item): void;
        up(event: KeyboardEvent): void;
        down(event: KeyboardEvent): void;
        generateHtml(): void;
        generateMenu(): void;
        post(x?: number, y?: number): void;
        unpostSubmenus(): void;
        unpost(): void;
        find(id: string): Item;
        protected parseItems(items: any[]): void;
        private parseItem(item);
    }
}
declare namespace ContextMenu {
    class MenuStore {
        protected store: HTMLElement[];
        private active;
        private menu;
        private counter;
        private attachedClass;
        private taborder;
        private attrMap;
        constructor(menu: ContextMenu);
        setActive(element: HTMLElement): void;
        getActive(): HTMLElement;
        next(): HTMLElement;
        previous(): HTMLElement;
        clear(): void;
        insert(element: HTMLElement): void;
        insert(elements: HTMLElement[]): void;
        insert(elements: NodeListOf<HTMLElement>): void;
        remove(element: HTMLElement): void;
        remove(element: HTMLElement[]): void;
        remove(element: NodeListOf<HTMLElement>): void;
        inTaborder(flag: boolean): void;
        insertTaborder(): void;
        removeTaborder(): void;
        private insertElement(element);
        private removeElement(element);
        private sort();
        private insertTaborder_();
        private removeTaborder_();
        private addTabindex(element);
        private removeTabindex(element);
        private addEvents(element);
        private addEvent(element, name, func);
        private removeEvents(element);
        private removeEvent(element, name, counter);
        private keydown(event);
    }
}
declare namespace ContextMenu {
    class ContextMenu extends AbstractMenu {
        private moving;
        private frame;
        private store_;
        private anchor;
        private widgets;
        static parse({menu: menu}: {
            menu: {
                pool: Array<Object>;
                items: Array<Object>;
                id: string;
            };
        }): ContextMenu;
        constructor();
        generateHtml(): void;
        protected display(): void;
        escape(event: KeyboardEvent): void;
        unpost(): void;
        left(event: KeyboardEvent): void;
        right(event: KeyboardEvent): void;
        getFrame(): HTMLElement;
        getStore(): MenuStore;
        post(): void;
        post(x: number, y: number): void;
        post(event: Event): void;
        post(element: HTMLElement): void;
        registerWidget(widget: Postable): void;
        unregisterWidget(widget: Postable): void;
        unpostWidgets(): void;
        private move_(next);
        private parseVariable({name: name, getter: getter, setter: setter});
    }
}
declare namespace ContextMenu {
    class SubMenu extends AbstractMenu {
        baseMenu: ContextMenu;
        private anchor;
        static parse({items: items, id: id}: {
            items: any[];
            id: string;
        }, anchor: Submenu): SubMenu;
        constructor(anchor: Submenu);
        getAnchor(): Submenu;
        post(): void;
        protected display(): void;
        private setBaseMenu();
    }
}
declare namespace ContextMenu {
    namespace MenuUtil {
        function close(item: Item): void;
        function getActiveElement(item: Item): HTMLElement;
        function error(error: Error, msg: string): void;
        function counter(): number;
    }
}
declare namespace ContextMenu {
    abstract class AbstractItem extends AbstractEntry implements Item {
        private _content;
        protected disabled: boolean;
        private id;
        private callbacks;
        constructor(menu: Menu, type: string, _content: string, id?: string);
        content: string;
        getId(): string;
        press(): void;
        protected executeAction(): void;
        registerCallback(func: Function): void;
        unregisterCallback(func: Function): void;
        mousedown(event: MouseEvent): void;
        mouseover(event: MouseEvent): void;
        mouseout(event: MouseEvent): void;
        generateHtml(): void;
        protected activate(): void;
        protected deactivate(): void;
        focus(): void;
        unfocus(): void;
        escape(event: KeyboardEvent): void;
        up(event: KeyboardEvent): void;
        down(event: KeyboardEvent): void;
        left(event: KeyboardEvent): void;
        right(event: KeyboardEvent): void;
        space(event: KeyboardEvent): void;
        disable(): void;
        enable(): void;
        private executeCallbacks_();
    }
}
declare namespace ContextMenu {
    abstract class AbstractVariableItem<T> extends AbstractItem implements VariableItem {
        protected span: HTMLElement;
        protected variable: Variable<T>;
        protected abstract generateSpan(): void;
        generateHtml(): void;
        register(): void;
        unregister(): void;
        update(): void;
        protected abstract updateAria(): void;
        protected abstract updateSpan(): void;
    }
}
declare namespace ContextMenu {
    namespace CssStyles {
        function addMenuStyles(opt_document: HTMLDocument): void;
        function addInfoStyles(opt_document: HTMLDocument): void;
    }
}
declare namespace ContextMenu {
    class CloseButton extends AbstractPostable {
        protected className: HtmlClass;
        protected role: string;
        private element;
        constructor(element: Postable);
        generateHtml(): void;
        protected display(): void;
        unpost(): void;
        keydown(event: KeyboardEvent): void;
        space(event: KeyboardEvent): void;
        mousedown(event: MouseEvent): void;
    }
}
declare namespace ContextMenu {
    class Info extends AbstractPostable {
        protected className: HtmlClass;
        protected role: string;
        private menu;
        private title;
        private signature;
        private contentDiv;
        private close;
        private content;
        constructor(title: string, content: Function, signature: string);
        attachMenu(menu: ContextMenu): void;
        getHtml(): HTMLElement;
        generateHtml(): void;
        post(): void;
        protected display(): void;
        click(event: MouseEvent): void;
        keydown(event: KeyboardEvent): void;
        escape(event: KeyboardEvent): void;
        unpost(): void;
        private generateClose();
        private generateTitle();
        private generateContent();
        private generateSignature();
    }
}
declare namespace ContextMenu {
    class Checkbox extends AbstractVariableItem<boolean> {
        protected role: string;
        static parse({content: content, variable: variable, id: id}: {
            content: string;
            variable: string;
            id: string;
        }, menu: Menu): Checkbox;
        constructor(menu: Menu, content: string, variable: string, id?: string);
        executeAction(): void;
        generateSpan(): void;
        protected updateAria(): void;
        updateSpan(): void;
    }
}
declare namespace ContextMenu {
    class Combo extends AbstractVariableItem<string> {
        protected role: string;
        private initial;
        private input;
        private inputEvent;
        static parse({content: content, variable: variable, id: id}: {
            content: string;
            variable: string;
            id: string;
        }, menu: Menu): Combo;
        constructor(menu: Menu, content: string, variable: string, id?: string);
        executeAction(): void;
        space(event: KeyboardEvent): void;
        focus(): void;
        generateHtml(): void;
        generateSpan(): void;
        inputKey(event: KeyboardEvent): void;
        keydown(event: KeyboardEvent): void;
        protected updateAria(): void;
        protected updateSpan(): void;
    }
}
declare namespace ContextMenu {
    class Command extends AbstractItem {
        private command;
        static parse({content: content, action: action, id: id}: {
            content: string;
            action: Function;
            id: string;
        }, menu: Menu): Command;
        constructor(menu: Menu, content: string, command: Function, id?: string);
        executeAction(): void;
    }
}
declare namespace ContextMenu {
    class Label extends AbstractItem {
        static parse({content: content, id: id}: {
            content: string;
            id: string;
        }, menu: Menu): Label;
        constructor(menu: Menu, content: string, id?: string);
        generateHtml(): void;
    }
}
declare namespace ContextMenu {
    class Radio extends AbstractVariableItem<string> {
        protected role: string;
        static parse({content: content, variable: variable, id: id}: {
            content: string;
            variable: string;
            id: string;
        }, menu: Menu): Radio;
        constructor(menu: Menu, content: string, variable: string, id?: string);
        executeAction(): void;
        generateSpan(): void;
        protected updateAria(): void;
        protected updateSpan(): void;
    }
}
declare namespace ContextMenu {
    class Rule extends AbstractEntry {
        protected className: HtmlClass;
        protected role: string;
        static parse({}: {}, menu: Menu): Rule;
        constructor(menu: Menu);
        generateHtml(): void;
        addEvents(element: HTMLElement): void;
    }
}
declare namespace ContextMenu {
    class Submenu extends AbstractItem {
        private span;
        private submenu;
        static parse({content: content, menu: submenu, id: id}: {
            content: string;
            menu: any;
            id: string;
        }, menu: Menu): Submenu;
        constructor(menu: Menu, content: string, id?: string);
        setSubmenu(menu: SubMenu): void;
        getSubmenu(): Menu;
        mouseover(event: MouseEvent): void;
        mouseout(event: MouseEvent): void;
        unfocus(): void;
        focus(): void;
        executeAction(): void;
        generateHtml(): void;
        left(event: KeyboardEvent): void;
        right(event: KeyboardEvent): void;
    }
}
declare namespace ContextMenu {
    class Popup extends AbstractPostable {
        private static popupSettings;
        private menu;
        private title;
        private content;
        private window;
        private localSettings;
        private windowList;
        private mobileFlag;
        private active;
        constructor(title: string, content: Function);
        attachMenu(menu: ContextMenu): void;
        post(): void;
        protected display(): void;
        unpost(): void;
        private generateContent();
        private resize();
    }
}
declare namespace ContextMenu {
    const TOUCH: {
        START: string;
        MOVE: string;
        END: string;
        CANCEL: string;
    };
    interface TouchNavigatable {
        touchstart(event: TouchEvent): void;
        touchmove(event: TouchEvent): void;
        touchend(event: TouchEvent): void;
        touchcancel(event: TouchEvent): void;
    }
}
