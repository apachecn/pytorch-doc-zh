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
 * @fileoverview    Configuration file for the tagformat package.
 *
 * @author dpvc@mathjax.org (Davide P. Cervone)
 */

import {Configuration} from '../Configuration.js';
import {TeX} from '../../tex.js';
import {AbstractTags, TagsFactory} from '../Tags.js';

/**
 * Number used to make tag class unique (each TeX input has to have its own because
 *  it needs access to the parse options)
 */
let tagID = 0;

/**
 * Configure a class to use for the tag handler that uses the input jax's options
 *   to control the formatting of the tags
 * @param {Configuration} config   The configuration for the input jax
 * @param {TeX} jax                The TeX input jax
 */
export function tagFormatConfig(config: Configuration, jax: TeX<any, any, any>) {

    /**
     * The original tag class to be extended (none, ams, or all)
     */
    const TagClass = TagsFactory.create(jax.parseOptions.options.tags).constructor as typeof AbstractTags;

    /**
     * A Tags object that uses the input jax options to perform the formatting
     *
     * Note:  We have to make a new class for each input jax since the format
     * methods don't have access to the input jax, and hence to its options.
     * If they did, we would use a common configTags class instead.
     */
    class Tagformat extends TagClass {

        /**
         * @override
         */
        public formatNumber(n: number) {
            return jax.parseOptions.options.tagFormat.number(n);
        }

        /**
         * @override
         */
        public formatTag(tag: string) {
            return jax.parseOptions.options.tagFormat.tag(tag);
        }

        /**
         * @override
         */
        public formatId(id: string) {
            return jax.parseOptions.options.tagFormat.id(id);
        }

        /**
         * @override
         */
        public formatUrl(id: string, base: string) {
            return jax.parseOptions.options.tagFormat.url(id, base);
        }
    }

    //
    //  Get a unique name for the tag class (since it is tied to the input jax)
    //  Note:  These never get freed, so they will accumulate if you create many
    //  TeX input jax instances with this extension.
    //
    tagID++;
    const tagName = 'configTags-' + tagID;
    //
    // Register the tag class
    //
    TagsFactory.add(tagName, Tagformat);
    jax.parseOptions.options.tags = tagName;
}

/**
 * The configuration object for configTags
 */
export const TagformatConfiguration = Configuration.create(
    'tagFormat', {
        config: tagFormatConfig,
        options: {
            tagFormat: {
                number: (n: number) => n.toString(),
                tag:    (tag: string) => '(' + tag + ')',
                id:     (id: string) => 'mjx-eqn-' + id.replace(/\s/g, '_'),
                url:    (id: string, base: string) => base + '#' + encodeURIComponent(id),
            }
        }
    }
);
