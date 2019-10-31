import './lib/tex-full.js';

import {registerTeX} from '../tex/register.js';
import {Loader} from '../../../../js/components/loader.js';
import {AllPackages} from '../../../../js/input/tex/AllPackages.js';
import '../../../../js/input/tex/require/RequireConfiguration.js';

Loader.preLoad(
    'input/tex-base',
    '[tex]/all-packages',
    '[tex]/require'
);

registerTeX(['require',...AllPackages]);
