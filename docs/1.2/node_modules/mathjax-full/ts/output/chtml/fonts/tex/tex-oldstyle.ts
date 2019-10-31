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
import {texOldstyle as font} from '../../../common/fonts/tex/tex-oldstyle.js';

export const texOldstyle: CHTMLCharMap = AddCSS(font, {
    0x20: {c: ' '},
    0x30: {c: '0'},
    0x31: {c: '1'},
    0x32: {c: '2'},
    0x33: {c: '3'},
    0x34: {c: '4'},
    0x35: {c: '5'},
    0x36: {c: '6'},
    0x37: {c: '7'},
    0x38: {c: '8'},
    0x39: {c: '9'},
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
    0x391: {c: 'A', f: ''},
    0x392: {c: 'B', f: ''},
    0x395: {c: 'E', f: ''},
    0x396: {c: 'Z', f: ''},
    0x397: {c: 'H', f: ''},
    0x399: {c: 'I', f: ''},
    0x39A: {c: 'K', f: ''},
    0x39C: {c: 'M', f: ''},
    0x39D: {c: 'N', f: ''},
    0x39F: {c: 'O', f: ''},
    0x3A1: {c: 'P', f: ''},
    0x3A2: {c: '\\398', f: ''},
    0x3A4: {c: 'T', f: ''},
    0x3A7: {c: 'X', f: ''},
    0x3D2: {c: '\\3A5', f: ''},
    0x3DC: {c: 'F', f: ''},
});
