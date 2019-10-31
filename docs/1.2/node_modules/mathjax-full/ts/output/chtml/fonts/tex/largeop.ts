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

import {CHTMLCharMap, AddCSS} from '../../FontData.js';
import {largeop as font} from '../../../common/fonts/tex/largeop.js';

export const largeop: CHTMLCharMap = AddCSS(font, {
    0x20: {c: ' '},
    0x28: {c: '('},
    0x29: {c: ')'},
    0x2F: {c: '/'},
    0x5B: {c: '['},
    0x5D: {c: ']'},
    0x7B: {c: '{'},
    0x7D: {c: '}'},
    0x2044: {c: '/'},
    0x2329: {c: '\\27E8'},
    0x232A: {c: '\\27E9'},
    0x2758: {c: '\\2223'},
    0x2A0C: {c: '\\222C\\222C'},
    0x3008: {c: '\\27E8'},
    0x3009: {c: '\\27E9'},
});
