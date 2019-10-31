
module.exports = {
    hooks: {
        "page": function(page) {
          page.content = page.content + '\n<script>console.log("plugin-popup....");document.onclick = function(e){ e.target.tagName === "IMG" && window.open(e.target.src,e.target.src)}</script><style>img{cursor:pointer}</style>';
          return page;
        }
    }
};