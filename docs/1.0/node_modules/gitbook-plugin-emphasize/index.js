
module.exports = {
    book: {
        assets: './assets',
        css: [
            'plugin.css'
        ]
    },
    blocks: {
        em: {
            process: function(block) {
                var type = block.kwargs.type || 'yellow';
                var color = block.kwargs.color;
                var style = '';

                if (color) {
                    style = 'background: '+color+';';
                }

                var body = ('<span class="pg-emphasize pg-emphasize-'+type+'" style="'+style+'">'
                    + (block.body)
                    + '</span>');

                return {
                    body: body,
                    parse: true
                };
            }
        }
    }
};
