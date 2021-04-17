(window.webpackJsonp=window.webpackJsonp||[]).push([[62],{465:function(e,t,a){"use strict";a.r(t);var s=a(26),n=Object(s.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"installation"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#installation"}},[e._v("#")]),e._v(" Installation")]),e._v(" "),a("p",[e._v("For the installation we recommend setting up a fresh "),a("a",{attrs:{href:"https://docs.conda.io/en/latest/miniconda.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("conda"),a("OutboundLink")],1),e._v(" environment:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("conda create -n biome python~"),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[e._v("3.7")]),e._v(".0 pip"),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v(">")]),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("=")]),a("span",{pre:!0,attrs:{class:"token number"}},[e._v("20.3")]),e._v(".0\nconda activate biome\n")])])]),a("p",[e._v("Once the conda environment is activated, you can install the latest release or the development version via pip.")]),e._v(" "),a("h2",{attrs:{id:"latest-release-recommended"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#latest-release-recommended"}},[e._v("#")]),e._v(" Latest release (recommended)")]),e._v(" "),a("p",[e._v("To install the latest release of "),a("em",[e._v("biome.text")]),e._v(" type in:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("pip "),a("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U biome-text\n")])])]),a("p",[e._v("After installing "),a("em",[e._v("biome.text")]),e._v(", the best way to test your installation is by running the "),a("em",[e._v("biome.text")]),e._v(" cli command:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("biome --help\n")])])]),a("p",[e._v("For the UI component to work you need a running "),a("a",{attrs:{href:"https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("Elasticsearch"),a("OutboundLink")],1),e._v(" instance.\nWe recommend running "),a("a",{attrs:{href:"https://www.elastic.co/guide/en/elasticsearch/reference/7.7/docker.html#docker-cli-run-dev-mode",target:"_blank",rel:"noopener noreferrer"}},[e._v("Elasticsearch via docker"),a("OutboundLink")],1),e._v(":")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("docker run -p "),a("span",{pre:!0,attrs:{class:"token number"}},[e._v("9200")]),e._v(":9200 -p "),a("span",{pre:!0,attrs:{class:"token number"}},[e._v("9300")]),e._v(":9300 -e "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"discovery.type=single-node"')]),e._v(" docker.elastic.co/elasticsearch/elasticsearch:7.3.2\n")])])]),a("h2",{attrs:{id:"master-branch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#master-branch"}},[e._v("#")]),e._v(" Master branch")]),e._v(" "),a("p",[e._v("The "),a("em",[e._v("master branch")]),e._v(" contains the latest features, but is less well tested.\nIf you are looking for a specific feature that has not been released yet, you can install the package from our master branch with:")]),e._v(" "),a("div",{staticClass:"language-shell script extra-class"},[a("pre",{pre:!0,attrs:{class:"language-shell"}},[a("code",[e._v("pip "),a("span",{pre:!0,attrs:{class:"token function"}},[e._v("install")]),e._v(" -U git+https://github.com/recognai/biome-text.git\n")])])]),a("p",[e._v("Be aware that the UI components will not work when installing the package this way.\nCheck out the "),a("RouterLink",{attrs:{to:"/documentation/community/3-developer_guides.html#setting-up-for-development"}},[e._v("developer guides")]),e._v(" on how to build the UI components manually.")],1)])}),[],!1,null,null,null);t.default=n.exports}}]);