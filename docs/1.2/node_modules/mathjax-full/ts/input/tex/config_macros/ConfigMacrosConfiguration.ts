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
 * @fileoverview    Configuration file for the config-macros package.
 *
 * @author dpvc@mathjax.org (Davide P. Cervone)
 */

import {Configuration} from '../Configuration.js';
import {expandable} from '../../../util/Options.js';
import {CommandMap} from '../SymbolMap.js';
import {Macro} from '../Symbol.js';
import NewcommandMethods from '../newcommand/NewcommandMethods.js';
import {TeX} from '../../tex.js';

/**
 * Create the user-defined macros from the macros option
 *
 * @param {Configuration} config   The configuration object for the input jax
 * @param {TeX} jax                The TeX input jax
 */
function configMacrosConfig(config: Configuration, jax: TeX<any, any, any>) {
    const macros = config.options.macros;
    for (const cs of Object.keys(macros)) {
        const def = (typeof macros[cs] === 'string' ? [macros[cs]] : macros[cs]);
        const macro = Array.isArray(def[2]) ?
            new Macro(cs, NewcommandMethods.MacroWithTemplate, def.slice(0,2).concat(def[2])) :
            new Macro(cs, NewcommandMethods.Macro, def);
        ConfigMacrosMap.add(cs, macro);
    }
}

/**
 * The command map for the autoloaded macros
 */
const ConfigMacrosMap = new CommandMap('configMacros', {}, {});

/**
 * The configuration object for configMacros
 */
export const ConfigMacrosConfiguration = Configuration.create(
    'configMacros', {
        handler: {macro: ['configMacros']},
        config: configMacrosConfig,
        options: {macros: expandable({})}
    }
);
