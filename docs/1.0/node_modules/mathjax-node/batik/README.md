If you intend to use MathJax-node's ability to create PNG images, you
should install [batik](http://xmlgraphics.apache.org/batik/download.html)
in this directory.  Just download it and unpack it here.  You need
`batik-rasterizer.jar` and the `lib` directory to be in the top level
of this directory. Since Batik v1.8 you have to either create a symlink
to `batik-rasterizer-1.8.jar` or change `mj-page.js` and `mj-single.js`
to point to the version-specific file name.
