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
import {texCalligraphic as font} from '../../../common/fonts/tex/tex-calligraphic.js';

export const texCalligraphic: CHTMLCharMap = AddCSS(font, {
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
    0x391: {c: 'A', f: 'I'},
    0x392: {c: 'B', f: 'I'},
    0x395: {c: 'E', f: 'I'},
    0x396: {c: 'Z', f: 'I'},
    0x397: {c: 'H', f: 'I'},
    0x399: {c: 'I', f: 'I'},
    0x39A: {c: 'K', f: 'I'},
    0x39C: {c: 'M', f: 'I'},
    0x39D: {c: 'N', f: 'I'},
    0x39F: {c: 'O', f: 'I'},
    0x3A1: {c: 'P', f: 'I'},
    0x3A2: {c: '\\398', f: 'I'},
    0x3A4: {c: 'T', f: 'I'},
    0x3A7: {c: 'X', f: 'I'},
    0x3D2: {c: '\\3A5', f: 'I'},
    0x3DC: {c: 'F', f: 'I'},
});
