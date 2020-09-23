(window.webpackJsonp=window.webpackJsonp||[]).push([[60],{431:function(e,t,s){"use strict";s.r(t);var a=s(26),n=Object(a.a)({},(function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[s("h1",{attrs:{id:"installation"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#installation"}},[e._v("#")]),e._v(" Installation")]),e._v(" "),s("p",[e._v("You can install "),s("em",[e._v("biome.text")]),e._v(" with pip or from source.\nFor the installation we recommend setting up a fresh "),s("a",{attrs:{href:"https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("conda environment"),s("OutboundLink")],1),e._v(":")]),e._v(" "),s("div",{staticClass:"language-shell extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[e._v("conda create -n biome "),s("span",{pre:!0,attrs:{class:"token assign-left variable"}},[e._v("python")]),s("span",{pre:!0,attrs:{class:"token operator"}},[e._v("==")]),s("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.7")]),e._v(".1\nconda activate biome\n")])])]),s("h2",{attrs:{id:"pip"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#pip"}},[e._v("#")]),e._v(" Pip")]),e._v(" "),s("p",[e._v("The recommended way for installing the library is using pip. You can install everything required for the library as follows:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[e._v("pip "),s("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" biome-text\n")])])]),s("h2",{attrs:{id:"from-source"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#from-source"}},[e._v("#")]),e._v(" From Source")]),e._v(" "),s("p",[e._v("If you want to contribute to "),s("em",[e._v("biome.text")]),e._v(" you have to install the library from source.\nClone the repository from github:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[e._v("git")]),e._v(" clone https://github.com/recognai/biome-text.git\n"),s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("cd")]),e._v(" biome-text\n")])])]),s("p",[e._v("and install the library in editable mode together with the test dependencies:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[e._v("pip "),s("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" --upgrade -e ."),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("[")]),e._v("testing"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("]")]),e._v("\n")])])]),s("p",[e._v("If the "),s("code",[e._v("make")]),e._v(" command is enabled in your system, you can also use the "),s("code",[e._v("make dev")]),e._v(" directive:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" dev\n")])])]),s("p",[e._v("For the UI to work you need to build the static web resources:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("cd")]),e._v(" ui \n"),s("span",{pre:!0,attrs:{class:"token function"}},[e._v("npm")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" \n"),s("span",{pre:!0,attrs:{class:"token function"}},[e._v("npm")]),e._v(" run build\n")])])]),s("p",[s("em",[e._v("Note: node>=12 is required in your machine.\nYou can follow installation instructions "),s("a",{attrs:{href:"https://nodejs.org/en/download/",target:"_blank",rel:"noopener noreferrer"}},[e._v("here"),s("OutboundLink")],1)])]),e._v(" "),s("p",[e._v("Again, you can also use the "),s("code",[e._v("make ui")]),e._v(" directive if the "),s("code",[e._v("make")]),e._v(" command is enabled in your system:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" ui\n")])])]),s("p",[e._v("You can see all defined directives with:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token function"}},[e._v("make")]),e._v(" "),s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("help")]),e._v("\n")])])]),s("p",[e._v("After installing "),s("em",[e._v("biome.text")]),e._v(", the best way to test your installation is by running the "),s("em",[e._v("biome.text")]),e._v(" cli command:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[e._v("biome --help\n")])])]),s("h2",{attrs:{id:"tests"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tests"}},[e._v("#")]),e._v(" Tests")]),e._v(" "),s("p",[s("em",[e._v("Biome.text")]),e._v(" uses "),s("a",{attrs:{href:"https://docs.pytest.org/en/latest/",target:"_blank",rel:"noopener noreferrer"}},[e._v("pytest"),s("OutboundLink")],1),e._v(" for its unit and integration tests.\nTo run the tests, make sure you installed "),s("em",[e._v("biome.text")]),e._v(" together with its test dependencies and simply execute pytest from within the "),s("code",[e._v("biome-text")]),e._v(" directory:")]),e._v(" "),s("div",{staticClass:"language-shell script extra-class"},[s("pre",{pre:!0,attrs:{class:"language-shell"}},[s("code",[s("span",{pre:!0,attrs:{class:"token builtin class-name"}},[e._v("cd")]),e._v(" biome-text\npytest\n")])])])])}),[],!1,null,null,null);t.default=n.exports}}]);