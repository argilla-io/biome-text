(window.webpackJsonp=window.webpackJsonp||[]).push([[56],{473:function(e,t,a){"use strict";a.r(t);var s=a(28),o=Object(s.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"developer-guides"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#developer-guides"}},[e._v("#")]),e._v(" Developer guides")]),e._v(" "),a("h2",{attrs:{id:"setting-up-for-development"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#setting-up-for-development"}},[e._v("#")]),e._v(" Setting up for development")]),e._v(" "),a("p",[e._v("To set up your system for "),a("em",[e._v("biome.text")]),e._v(" development, you first of all have to "),a("a",{attrs:{href:"https://guides.github.com/activities/forking/",target:"_blank",rel:"noopener noreferrer"}},[e._v("fork"),a("OutboundLink")],1),e._v("\nour repository and clone your fork to your computer:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("git")]),e._v(" clone https://github.com/"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("[")]),e._v("your-github-username"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("]")]),e._v("/biome-text.git\n"),a("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("cd")]),e._v(" biome-text\n")])])]),a("p",[e._v("To keep your fork's master branch up to date with our repo you should add it as an "),a("a",{attrs:{href:"https://dev.to/louhayes3/git-add-an-upstream-to-a-forked-repo-1mik",target:"_blank",rel:"noopener noreferrer"}},[e._v("upstream remote branch"),a("OutboundLink")],1),e._v(":")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("git")]),e._v(" remote "),a("span",{pre:!0,attrs:{class:"token function"}},[e._v("add")]),e._v(" upstream https://github.com/recognai/biome-text.git\n")])])]),a("p",[e._v("Now go ahead and create a new conda environment in which the development will take place and activate it:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("conda "),a("span",{pre:!0,attrs:{class:"token function"}},[e._v("env")]),e._v(" create -f environment_dev.yml\nconda activate biometext\n")])])]),a("p",[e._v("Once you activated the conda environment, it is time to install "),a("em",[e._v("biome.text")]),e._v(" in editable mode with all its development dependencies.\nThe best way to do this is to take advantage of the make directive:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" dev\n")])])]),a("p",[e._v("After installing "),a("em",[e._v("biome.text")]),e._v(", the best way to test your installation is by running the "),a("em",[e._v("biome.text")]),e._v(" cli command:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("biome --help\n")])])]),a("h3",{attrs:{id:"running-tests-locally"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#running-tests-locally"}},[e._v("#")]),e._v(" Running tests locally")]),e._v(" "),a("p",[a("em",[e._v("Biome.text")]),e._v(" uses "),a("a",{attrs:{href:"https://docs.pytest.org/en/latest/",target:"_blank",rel:"noopener noreferrer"}},[e._v("pytest"),a("OutboundLink")],1),e._v(" for its unit and integration tests.\nIf you are working on the code base we advise you to run our tests locally before submitting a Pull Request (see below) to make sure your changes did not break and existing functionality.\nTo achieve this you can simply run:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("test")]),e._v("\n")])])]),a("p",[e._v("If you open a Pull Request, the test suite will be run automatically via a GitHub Action.")]),e._v(" "),a("h3",{attrs:{id:"serving-docs-locally"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#serving-docs-locally"}},[e._v("#")]),e._v(" Serving docs locally")]),e._v(" "),a("p",[e._v("If you are working on the documentation and want to check out the results locally on your machine, you can simply run:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" docs\n")])])]),a("p",[e._v("The docs will be built and deployed automatically via a GitHub Action when our master branch is updated.\nIf for some reason you want to build them locally, you can do so with:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" build_docs\n")])])]),a("h2",{attrs:{id:"make-a-release"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#make-a-release"}},[e._v("#")]),e._v(" Make a release")]),e._v(" "),a("p",[e._v("To make a release you have to follow 4 steps:")]),e._v(" "),a("ol",[a("li",[a("p",[e._v("Run the "),a("code",[e._v("prepare_versioned_build.sh")]),e._v(" script inside the "),a("code",[e._v("docs")]),e._v(' folder and commit the changes to the master branch.\nThe commit message should say something like: "v2.2.0 release".')])]),e._v(" "),a("li",[a("p",[e._v("Create a new "),a("a",{attrs:{href:"https://docs.github.com/en/free-pro-team@latest/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release",target:"_blank",rel:"noopener noreferrer"}},[e._v("GitHub release"),a("OutboundLink")],1),e._v(".")]),e._v(" "),a("p",[e._v("The version tags should be "),a("code",[e._v("v1.1.0")]),e._v(" or for release candidates "),a("code",[e._v("v1.1.0rc1")]),e._v(".\nMajor and minor releases should always be made against the master branch, bugfix releases against the corresponding minor release tag.")]),e._v(" "),a("p",[e._v("After publishing the release, the CI is triggered and if everything goes well the release gets published on PyPi.\nThe CI does:")]),e._v(" "),a("ul",[a("li",[e._v("run tests & build docs")]),e._v(" "),a("li",[e._v("build package")]),e._v(" "),a("li",[e._v("upload to testpypi")]),e._v(" "),a("li",[e._v("install from testpypi")]),e._v(" "),a("li",[e._v("upload to pypi")])])]),e._v(" "),a("li",[a("p",[e._v('Revert the last commit in which you changed the docs, the commit message should read something like:\n"back to master release".')])]),e._v(" "),a("li",[a("p",[a("strong",[e._v("Docs")]),e._v(": In order for the Algolia Search to work, you need to add the new version number of the docs to our\nalgolia "),a("a",{attrs:{href:"https://github.com/algolia/docsearch-configs/blob/master/configs/recogn_biome-text.json",target:"_blank",rel:"noopener noreferrer"}},[e._v("config file"),a("OutboundLink")],1),e._v(" and submit a PR.")])])]),e._v(" "),a("p",[e._v("Under the hood the versioning of our package is managed by "),a("a",{attrs:{href:"https://github.com/pypa/setuptools_scm",target:"_blank",rel:"noopener noreferrer"}},[a("code",[e._v("setuptools_scm")]),a("OutboundLink")],1),e._v(",\nthat basically works with the git tags in a repo.")])])}),[],!1,null,null,null);t.default=o.exports}}]);