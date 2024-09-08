## Fireship NX Quickstart tutorial

* [video](https://www.youtube.com/watch?v=VUyBY72mwrQ)

````bash
npx create-nx-workspace fireship-tutorial --package-manager=yarn

# ✔ Which stack do you want to use? · none
# ✔ Package-based monorepo, integrated monorepo, or standalone project? · package-based
#
# ✔ Which CI provider would you like to use? · skip
# ✔ Would you like remote caching to make your build faster? · skip

# install angular
yarn global add @angular/cli
````

Second method (integrated project) for comparison:
````bash
$ npx create-nx-workspace fireship-tut --package-manager=yarn

#  NX   Let's create a new workspace [https://nx.dev/getting-started/intro]
# 
# ✔ Which stack do you want to use? · none
# ✔ Package-based monorepo, integrated monorepo, or standalone project? · integrated
# 
# ✔ Which CI provider would you like to use? · skip
# ✔ Would you like remote caching to make your build faster? · skip
# 
#  NX   Creating your v19.6.5 workspace.
# 
# ✔ Installing dependencies with yarn
# ✔ Successfully created the workspace: fireship-tut.
````

version 3 with angular built in:

````bash
$ npx create-nx-workspace fireship-tut-angular --package-manager=yarn

#  NX   Let's create a new workspace [https://nx.dev/getting-started/intro]
#
# ✔ Which stack do you want to use? · angular
# ✔ Integrated monorepo, or standalone project? · integrated
# ✔ Application name · fireship-tut-angular
# ✔ Which bundler would you like to use? · esbuild
# ✔ Default stylesheet format · css
# ✔ Do you want to enable Server-Side Rendering (SSR) and Static Site Generation (SSG/Prerendering)? · Yes
#
# ✔ Test runner to use for end to end (E2E) tests · playwright
# ✔ Which CI provider would you like to use? · skip
# ✔ Would you like remote caching to make your build faster? · skip
````