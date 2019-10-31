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
import {texVariant as font} from '../../../common/fonts/tex/tex-variant.js';

export const texVariant: CHTMLCharMap = AddCSS(font, {
    0x20: {c: ' '},
    0x41: {c: 'A'},
    0x42: {c: 'B'},
    0x43: {c: 'C'},
    0x44: {c: 'D'},
    0x45: {c: 'E'},
    0x46: {c: 'F'},
    0x47: {c: 'G'},
    0x48: {c: 'H'},
    0x49: {c: 'I'},
    0x4A: {c: 'J'},
    0x4B: {c: 'K'},
    0x4C: {c: 'L'},
    0x4D: {c: 'M'},
    0x4E: {c: 'N'},
    0x4F: {c: 'O'},
    0x50: {c: 'P'},
    0x51: {c: 'Q'},
    0x52: {c: 'R'},
    0x53: {c: 'S'},
    0x54: {c: 'T'},
    0x55: {c: 'U'},
    0x56: {c: 'V'},
    0x57: {c: 'W'},
    0x58: {c: 'X'},
    0x59: {c: 'Y'},
    0x5A: {c: 'Z'},
    0x6B: {c: 'k'},
    0x3DC: {c: '\\E008'},
    0x3F0: {c: '\\E009'},
    0x210F: {f: ''},
    0x2216: {f: ''},
    0x2224: {c: '\\E006'},
    0x2226: {c: '\\E007'},
    0x2268: {c: '\\E00C'},
    0x2269: {c: '\\E00D'},
    0x2270: {c: '\\E011'},
    0x2271: {c: '\\E00E'},
    0x2288: {c: '\\E016'},
    0x2289: {c: '\\E018'},
    0x228A: {c: '\\E01A'},
    0x228B: {c: '\\E01B'},
    0x2A87: {c: '\\E010'},
    0x2A88: {c: '\\E00F'},
    0x2ACB: {c: '\\E017'},
    0x2ACC: {c: '\\E019'},
});
