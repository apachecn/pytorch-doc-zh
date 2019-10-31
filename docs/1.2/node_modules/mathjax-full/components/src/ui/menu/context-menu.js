import '../../../../../mj-context-menu/dist/context_menu.js';

//
//  Check if the global ContextMenu is not defined (in node it won't be
//  because it is local to the module), and define dummy classes
//  for the ones we need (they should never be used in node since the
//  menus are only useful in the browser).
//
if (typeof ContextMenu === 'undefined') {
    global.ContextMenu = {
        ContextMenu: class {},
        Info: class {}
    };
}
