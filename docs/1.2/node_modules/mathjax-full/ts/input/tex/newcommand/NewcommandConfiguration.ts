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
 * @fileoverview Configuration file for the Newcommand package.
 *
 * @author v.sorge@mathjax.org (Volker Sorge)
 */

import {Configuration} from '../Configuration.js';
import {BeginEnvItem} from './NewcommandItems.js';
import {ExtensionMaps} from '../MapHandler.js';
import './NewcommandMappings.js';


/**
 * Init method for Newcommand package.
 * @param {Configuration} config The current configuration.
 */
let init = function(config: Configuration) {
    if (config.handler['macro'].indexOf(ExtensionMaps.NEW_COMMAND) < 0) {
        config.append(Configuration.extension());
    }
};


export const NewcommandConfiguration = Configuration.create(
  'newcommand',
  {
    handler: {
      macro: ['Newcommand-macros']
    },
    items: {
      [BeginEnvItem.prototype.kind]: BeginEnvItem,
    },
    options: {maxMacros: 1000},
    init: init
  }
);


