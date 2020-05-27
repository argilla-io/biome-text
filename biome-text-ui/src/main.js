import Vue from 'vue';
import SvgIcon from 'vue-svgicon';
import vClickOutside from 'v-click-outside';
import VueVirtualScroller from 'vue-virtual-scroller';
import vueSmoothScroll from 'vue-smooth-scroll';
import VueMoment from 'vue-moment';
import VueVega from 'vue-vega';
import router from './router';
import App from './App';

Vue.use(SvgIcon);
Vue.use(vClickOutside);
Vue.use(VueVirtualScroller);
Vue.use(VueMoment);
Vue.use(VueVega);
Vue.use(vueSmoothScroll);

Vue.config.productionTip = false;

new Vue({
  router,
  render: h => h(App),
}).$mount('#app');
