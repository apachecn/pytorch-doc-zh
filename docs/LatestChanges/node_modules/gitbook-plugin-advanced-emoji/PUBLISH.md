# Howto Publish to npmjs.org

use this because of npm proxy:

```
npm --registry https://registry.npmjs.org/ login
npm --registry https://registry.npmjs.org/ publish
```

Then tag

```
tag -a 0.1.6 -m "rel 0.1.6"
git push origin 0.1.6
```

Go to github releases and create release from tag.

Increase version in package.json.