var fs = require('fs')
var crypto = require('crypto')
var request = require('sync-request')

var doc_dir = '../docs'
var img_dir = doc_dir + '/img'

try { fs.mkdirSync(img_dir) }
catch(e) {}

function processTex(md) {
    
    var rm;
    while(rm = /\$(.+?)\$/g.exec(md)){
        var tex = rm[1]
        //console.log(rm)
        var url = 'http://latex.codecogs.com/gif.latex?'
            + encodeURIComponent(tex)
        var tex_md5 = crypto.createHash("md5").update(tex).digest('hex')
        var img = request('get', url).getBody()
        
        // replace_all
        md = md.split(rm[0]).join(`![${tex}](img/tex-${tex_md5}.gif)`)
        fs.writeFileSync(`${img_dir}/tex-${tex_md5}.gif`, img)
        
        console.log(tex_md5)
    }
    
    return md
    
}

function main() {

    var flist = fs.readdirSync(doc_dir).filter(s => s.endsWith('.md'))

    for(var fname of flist) {
        
        fname = doc_dir + '/' + fname
        console.log(fname)
        
        var md = fs.readFileSync(fname, 'utf-8')
        md = processTex(md)    
        fs.writeFileSync(fname, md)
    }

    console.log('done')

}

main()